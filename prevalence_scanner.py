#!/usr/bin/env python3
"""
Prevalence Scanner — batch ICH inference over a DICOM study archive.

Designed to run unattended over an institution's full CT archive — including
studies that are not head CTs.  Non-head studies are rejected cheaply with a
two-stage filter before any inference is attempted:

  Stage 1 — Fast study pre-filter (single DICOM file, ~1 ms per study):
    Read one DICOM header to check Modality and StudyDescription.
    Any study that is not CT, or whose description contains no brain/head
    keywords, is discarded immediately without reading the full folder.

  Stage 2 — Series selector (full folder headers, no pixel data):
    Read all series headers and apply the NCCT selection criteria
    (no contrast, correct image type, thinnest qualifying slice).
    Studies with no qualifying series are rejected and counted separately.

  Stage 3 — Inference (MaxViT, GPU):
    Run the trained model on the selected series.

Date filtering uses DICOM StudyDate from the header, not filesystem mtime,
so it works correctly on archived or migrated data.

A scan-progress file records folders already visited (including rejected
ones) so interrupted multi-day runs can resume without re-checking
previously discarded studies.

Directory layouts supported
----------------------------
  PACS-style (most common):
    archive_root/<patient>/<study_uid>/<series_uid>/*.dcm

  Date-organised:
    archive_root/YYYY/MM/DD/<study_folder>/*.dcm

  Flat per-study:
    archive_root/<study_folder>/*.dcm

  Flat (all DCM in one directory — treated as one study):
    archive_root/*.dcm

Usage
-----
    # First run: full archive, skip already-processed
    python prevalence_scanner.py /pacs/archive/ct

    # Incremental update: only new studies since last run
    python prevalence_scanner.py /pacs/archive/ct --days 30

    # Dry run: count and classify studies without inference
    python prevalence_scanner.py /pacs/archive/ct --dry-run

    # Summary report after scanning
    python prevalence_scanner.py /pacs/archive/ct --report

    # Report and trend only (no scanning)
    python prevalence_scanner.py --report-only --trend month

    # Parallel header reading (default 8 workers)
    python prevalence_scanner.py /pacs/archive/ct --io-workers 16
"""

import os
import sys
import json
import time
import argparse
import warnings
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prevalence_db import PrevalenceDB, normalize_location
from ich_inference import run_inference

# ── Head-CT pre-filter keywords ───────────────────────────────────────────────

# Any of these in StudyDescription → candidate head CT (go to stage 2)
HEAD_STUDY_KEYWORDS = {
    "head", "brain", "cranial", "cranium", "intracranial",
    "neuro", "skull", "cerebr", "cerebell",
    "ncct", "nchct", "ct head", "head ct",
}

# Any of these in StudyDescription → definitely not head CT (reject immediately)
NON_HEAD_STUDY_KEYWORDS = {
    "chest", "thorax", "thoracic", "abdomen", "abdominal",
    "pelvis", "pelvic", "spine", "lumbar", "cervical spine",
    "thoracic spine", "extremit", "shoulder", "knee", "ankle",
    "wrist", "elbow", "hip", "femur", "tibia", "cardiac",
    "coronary", "cardiac ct", "aorta", "renal", "hepat",
    "colono", "dental",
}

# NCCT series selection keywords (same as ich_agent.py system prompt)
NCCT_DESCRIPTION_KEYWORDS = {
    "head", "brain", "axial", "non-con", "noncon", "without",
    "ncct", "w/o", "wo", "plain", "unenhanced",
}

EXCLUDE_DESCRIPTION_KEYWORDS = {
    "perfusion", "angio", "cta", "ctp", "scout", "localizer",
    "mip", "reformat", "cor", "sag", "3d", "bone", "recon",
}


# ── Stage 1: fast study pre-filter ────────────────────────────────────────────

def _fast_study_check(study_folder: Path) -> tuple[bool, str, str, str]:
    """
    Read ONE DICOM file from the study folder (the first one found).
    Returns (is_candidate, study_date_iso, study_uid, reject_reason).

    is_candidate = True  → proceed to full header read (stage 2)
    is_candidate = False → skip the whole study folder
    """
    import pydicom

    dcm_file = next(study_folder.rglob("*.dcm"), None)
    if dcm_file is None:
        return False, "", "", "no_dcm_files"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
    except Exception:
        return False, "", "", "header_read_error"

    # Must be CT modality
    modality = str(getattr(ds, "Modality", "") or "").upper().strip()
    if modality and modality != "CT":
        return False, "", "", f"wrong_modality_{modality}"

    # Parse DICOM StudyDate → ISO format
    raw_date = str(getattr(ds, "StudyDate", "") or "")
    study_date = ""
    if len(raw_date) == 8 and raw_date.isdigit():
        study_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"

    study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")

    # Check StudyDescription for head CT keywords
    study_desc = str(getattr(ds, "StudyDescription", "") or "").lower()

    # Explicit non-head keyword → fast reject
    for kw in NON_HEAD_STUDY_KEYWORDS:
        if kw in study_desc:
            return False, study_date, study_uid, f"non_head_desc"

    # Positive head keyword → candidate
    for kw in HEAD_STUDY_KEYWORDS:
        if kw in study_desc:
            return True, study_date, study_uid, ""

    # StudyDescription is empty or ambiguous — let stage 2 decide
    # (series descriptions may be more informative)
    if not study_desc.strip():
        return True, study_date, study_uid, ""

    # Description present but no head keyword and no non-head keyword
    # e.g. "CT WITHOUT CONTRAST" — could be anything; pass to stage 2
    return True, study_date, study_uid, ""


# ── Stage 2: series selector ──────────────────────────────────────────────────

def _description_score(desc: str) -> int:
    d = desc.lower()
    score  = sum(1 for kw in NCCT_DESCRIPTION_KEYWORDS    if kw in d)
    score -= sum(2 for kw in EXCLUDE_DESCRIPTION_KEYWORDS if kw in d)
    return score


def select_ncct_series(series_list: list[dict]) -> Optional[dict]:
    """
    Rule-based axial NCCT head series selector.
    Returns the best qualifying series dict, or None.
    """
    candidates = []
    for s in series_list:
        if s.get("modality", "").upper() != "CT":
            continue
        contrast = s.get("contrast_bolus_agent", "").strip()
        if contrast and contrast.lower() not in ("", "none", "n/a"):
            continue
        image_type = s.get("image_type", [])
        if image_type and str(image_type[0]).upper() == "DERIVED":
            continue
        if s.get("slice_count", 0) < 10:
            continue
        desc  = s.get("series_description", "")
        score = _description_score(desc)
        if score <= 0 and desc.strip():
            continue
        try:
            thickness = float(s.get("slice_thickness", "99") or "99")
        except (ValueError, TypeError):
            thickness = 99.0
        candidates.append((score, thickness, s))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][2]


# ── Stage 2: full header read ─────────────────────────────────────────────────

def _read_study_header(study_folder: Path) -> dict:
    """
    Read all DICOM series headers (no pixel data) from a study folder.
    Returns header dict suitable for select_ncct_series(), or {}.
    """
    import pydicom

    series: dict[str, dict] = {}
    first_ds = None

    for dcm_file in sorted(study_folder.rglob("*.dcm")):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            if first_ds is None:
                first_ds = ds
            uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
            if uid not in series:
                series[uid] = {
                    "series_uid":           uid,
                    "series_number":        str(getattr(ds, "SeriesNumber",        "")),
                    "series_description":   str(getattr(ds, "SeriesDescription",   "")),
                    "modality":             str(getattr(ds, "Modality",            "")),
                    "image_type":           list(getattr(ds, "ImageType",          [])),
                    "slice_thickness":      str(getattr(ds, "SliceThickness",      "")),
                    "contrast_bolus_agent": str(getattr(ds, "ContrastBolusAgent",  "")),
                    "study_uid":            str(getattr(ds, "StudyInstanceUID",    "")),
                    "patient_id":           str(getattr(ds, "PatientID",           "")),
                    "series_folder":        str(dcm_file.parent),
                    "slice_count":          0,
                }
            series[uid]["slice_count"] += 1
        except Exception:
            continue

    if not series or first_ds is None:
        return {}

    raw_date = str(getattr(first_ds, "StudyDate", "") or "")
    raw_time = str(getattr(first_ds, "StudyTime", "") or "").split(".")[0]
    study_date = ""
    study_dt   = ""
    if len(raw_date) == 8 and raw_date.isdigit():
        study_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
        if len(raw_time) >= 6:
            study_dt = (f"{study_date}T"
                        f"{raw_time[:2]}:{raw_time[2:4]}:{raw_time[4:6]}")
        else:
            study_dt = study_date

    first_series = next(iter(series.values()))
    return {
        "study_uid":      first_series.get("study_uid", str(study_folder)),
        "patient_id":     first_series.get("patient_id", ""),
        "exam_date":      study_date,
        "exam_datetime":  study_dt,
        "raw_department": str(getattr(first_ds, "InstitutionalDepartmentName", "") or ""),
        "series_list":    list(series.values()),
    }


# ── Study folder discovery ────────────────────────────────────────────────────

def _study_folders(root: Path) -> list[Path]:
    """
    Find candidate study-level folders under root.

    Heuristic: a folder is study-level if it contains .dcm files directly
    OR if its subdirectories contain .dcm files (series-level layout).
    Each unique study folder is returned once.
    """
    folders: set[Path] = set()
    for dcm in root.rglob("*.dcm"):
        p = dcm.parent
        # If the parent of this folder also has subdirectories that contain
        # DCM files, p.parent is the study folder; otherwise p is.
        try:
            sibling_has_dcm = any(
                c.is_dir() and next(c.glob("*.dcm"), None) is not None
                for c in p.parent.iterdir()
                if c != p and c.is_dir()
            )
        except PermissionError:
            sibling_has_dcm = False
        folders.add(p.parent if sibling_has_dcm else p)
    return sorted(folders)


# ── Scan-progress file (resume support) ──────────────────────────────────────

class ScanProgress:
    """
    Lightweight file-based record of folders already visited.
    Allows interrupted archive runs to resume without re-checking
    folders that were already rejected at the pre-filter stage.

    Format: one absolute folder path per line.
    """

    def __init__(self, path: Path):
        self.path = path
        self._visited: set[str] = set()
        if path.exists():
            with open(path) as f:
                self._visited = {line.strip() for line in f if line.strip()}

    def seen(self, folder: Path) -> bool:
        return str(folder) in self._visited

    def mark(self, folder: Path):
        key = str(folder)
        if key not in self._visited:
            self._visited.add(key)
            with open(self.path, "a") as f:
                f.write(key + "\n")

    def __len__(self):
        return len(self._visited)


# ── Worker: pre-filter + header read (runs in thread pool) ───────────────────

def _worker_process_folder(
    folder:        Path,
    date_cutoff:   Optional[str],   # ISO date string "YYYY-MM-DD" or None
    existing_uids: set[str],
    skip_existing: bool,
) -> dict:
    """
    Process one study folder through stages 1 and 2.
    Returns a result dict consumed by the main loop.
    Keys: status, folder, study_uid, exam_date, header (if status=="ready")
    """
    # ── Stage 1: fast pre-filter ──────────────────────────────────────────────
    try:
        is_candidate, study_date, study_uid, reject_reason = _fast_study_check(folder)
    except Exception as exc:
        return {"status": "error", "folder": folder, "reason": str(exc)}

    if not is_candidate:
        return {"status": "not_head_ct", "folder": folder, "reason": reject_reason}

    # ── Date filter (uses DICOM StudyDate) ────────────────────────────────────
    if date_cutoff and study_date:
        if study_date < date_cutoff:
            return {"status": "date_filtered", "folder": folder,
                    "study_date": study_date}

    # ── Skip already-processed ────────────────────────────────────────────────
    if skip_existing and study_uid and study_uid in existing_uids:
        return {"status": "already_processed", "folder": folder,
                "study_uid": study_uid}

    # ── Stage 2: full header read ─────────────────────────────────────────────
    try:
        header = _read_study_header(folder)
    except Exception as exc:
        return {"status": "error", "folder": folder, "reason": str(exc)}

    if not header:
        return {"status": "error", "folder": folder, "reason": "empty_header"}

    # Confirm date filter with more reliable header date (may differ from fast check)
    if date_cutoff and header.get("exam_date"):
        if header["exam_date"] < date_cutoff:
            return {"status": "date_filtered", "folder": folder,
                    "study_date": header["exam_date"]}

    # ── Stage 2: series selection ─────────────────────────────────────────────
    series = select_ncct_series(header["series_list"])
    if series is None:
        return {"status": "no_ncct_series", "folder": folder,
                "study_uid": header.get("study_uid", "")}

    return {"status": "ready", "folder": folder,
            "header": header, "series": series}


# ── Main scan function ────────────────────────────────────────────────────────

def scan_archive(
    root_dir:        str | Path,
    db:              PrevalenceDB,
    checkpoint_path: str = "",
    days_back:       Optional[int] = None,
    skip_existing:   bool = True,
    dry_run:         bool = False,
    io_workers:      int  = 8,
    verbose:         bool = True,
    progress_file:   Optional[Path] = None,
) -> dict:
    """
    Scan a DICOM study archive and populate the prevalence database.

    Parameters
    ----------
    root_dir        : archive root directory (any layout)
    db              : PrevalenceDB instance
    checkpoint_path : MaxViT .pth file
    days_back       : only process studies with DICOM StudyDate >= today - N days
    skip_existing   : skip study_uids already in the DB
    dry_run         : run stages 1 & 2 but skip inference; print statistics
    io_workers      : thread-pool size for parallel header reading
    progress_file   : path to scan-progress file for resume support
                      (default: <db_dir>/scan_progress.txt)

    Returns
    -------
    dict: n_found, n_not_head_ct, n_date_filtered, n_already_processed,
          n_no_ncct_series, n_processed, n_positive, n_error, elapsed_seconds,
          slices_per_second
    """
    if not checkpoint_path:
        checkpoint_path = str(
            Path(__file__).parent / "checkpoints_maxvit" / "best_maxvit_ich.pth"
        )

    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Archive root not found: {root_dir}")

    if progress_file is None:
        progress_file = db.db_path.parent / "scan_progress.txt"

    date_cutoff = None
    if days_back:
        date_cutoff = (datetime.now() - timedelta(days=days_back)).date().isoformat()

    t0 = time.perf_counter()

    if verbose:
        print(f"\n{'='*68}")
        print(f"ICH Prevalence Scanner")
        print(f"  Archive      : {root_dir}")
        print(f"  Date cutoff  : {date_cutoff or 'none (all studies)'}")
        print(f"  DB           : {db.db_path}")
        print(f"  Progress file: {progress_file}")
        print(f"  IO workers   : {io_workers}")
        print(f"  Dry run      : {dry_run}")
        print(f"{'='*68}")

    # ── Discover folders ──────────────────────────────────────────────────────
    if verbose:
        print("Discovering study folders...", end="", flush=True)
    all_folders = _study_folders(root_dir)
    if verbose:
        print(f" {len(all_folders):,} candidate folders")

    # ── Load resume state ─────────────────────────────────────────────────────
    progress = ScanProgress(progress_file)
    folders  = [f for f in all_folders if not progress.seen(f)]
    if verbose and len(all_folders) != len(folders):
        print(f"  Resuming: {len(all_folders) - len(folders):,} folders already "
              f"visited in previous runs, {len(folders):,} remaining")

    # ── Load existing study UIDs ──────────────────────────────────────────────
    existing_uids: set[str] = set()
    if skip_existing:
        with db._conn() as conn:
            rows = conn.execute("SELECT study_uid FROM study_results").fetchall()
        existing_uids = {r["study_uid"] for r in rows}
        if verbose:
            print(f"  {len(existing_uids):,} studies already in DB")

    counters = {
        "n_found":            len(folders),
        "n_not_head_ct":      0,
        "n_date_filtered":    0,
        "n_already_processed": 0,
        "n_no_ncct_series":   0,
        "n_processed":        0,
        "n_positive":         0,
        "n_error":            0,
        "total_slices":       0,
    }

    if len(folders) == 0:
        if verbose:
            print("  Nothing to do.\n")
        counters["elapsed_seconds"]   = 0.0
        counters["slices_per_second"] = 0.0
        return counters

    if verbose:
        print(f"\nProcessing {len(folders):,} folders "
              f"({'dry run — no inference' if dry_run else 'with inference'})...\n")

    # ── Process folders via thread pool (stages 1 & 2) → inference (stage 3) ─
    inference_queue: list[dict] = []  # items with status=="ready"

    def _submit_batch(batch):
        """Run pre-filter + header read on a batch in parallel."""
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=io_workers) as ex:
            futures = {
                ex.submit(
                    _worker_process_folder,
                    f, date_cutoff, existing_uids, skip_existing
                ): f
                for f in batch
            }
            for fut in concurrent.futures.as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as exc:
                    results.append({"status": "error",
                                    "folder": futures[fut],
                                    "reason": str(exc)})
        return results

    # Process in chunks so we can interleave I/O batches with GPU inference
    CHUNK = 256   # folders per I/O batch
    done  = 0

    for chunk_start in range(0, len(folders), CHUNK):
        chunk = folders[chunk_start : chunk_start + CHUNK]

        # Parallel stage 1+2 for this chunk
        worker_results = _submit_batch(chunk)

        for res in worker_results:
            folder = res["folder"]
            status = res["status"]

            progress.mark(folder)   # record as visited regardless of outcome

            if status == "not_head_ct":
                counters["n_not_head_ct"] += 1
            elif status == "date_filtered":
                counters["n_date_filtered"] += 1
            elif status == "already_processed":
                counters["n_already_processed"] += 1
            elif status == "no_ncct_series":
                counters["n_no_ncct_series"] += 1
            elif status == "error":
                counters["n_error"] += 1
                if verbose:
                    print(f"  ERROR {folder.name}: {res.get('reason','')}")
            elif status == "ready":
                if dry_run:
                    # Count it but don't infer
                    counters["n_processed"] += 1
                else:
                    inference_queue.append(res)

        done += len(chunk)
        if verbose and not dry_run:
            pct = done / len(folders) * 100
            ready = len(inference_queue)
            print(f"  Prepared {done:>7,}/{len(folders):,} ({pct:.0f}%)  "
                  f"queued for inference: {ready:,}", end="\r", flush=True)

        # ── Stage 3: inference (drain the queue as it fills) ──────────────────
        if not dry_run and (len(inference_queue) >= 16 or
                            chunk_start + CHUNK >= len(folders)):
            if verbose and inference_queue:
                print()  # newline after \r progress
            for item in inference_queue:
                header = item["header"]
                series = item["series"]
                study_uid = header["study_uid"]

                if verbose:
                    dept = normalize_location(header.get("raw_department", ""))
                    print(f"  {study_uid[:24]}  {dept:<14}  "
                          f"{series.get('series_description','')[:28]}",
                          end="  ", flush=True)

                try:
                    result = run_inference(
                        series_folder   = series["series_folder"],
                        checkpoint_path = checkpoint_path,
                        verbose         = False,
                    )
                except Exception as exc:
                    if verbose:
                        print(f"INFERENCE ERROR: {exc}")
                    counters["n_error"] += 1
                    continue

                if "error" in result:
                    if verbose:
                        print(f"ERROR: {result['error']}")
                    counters["n_error"] += 1
                    continue

                db.record_study(
                    study_uid         = study_uid,
                    ai_positive       = result["overall_positive"],
                    patient_id        = header.get("patient_id", ""),
                    exam_date         = header.get("exam_date", ""),
                    exam_datetime     = header.get("exam_datetime", ""),
                    raw_department    = header.get("raw_department", ""),
                    dominant_class    = result.get("dominant_class", ""),
                    study_level_probs = result.get("study_level", {}),
                    series_folder     = series["series_folder"],
                    checkpoint        = checkpoint_path,
                )
                existing_uids.add(study_uid)

                counters["n_processed"] += 1
                counters["total_slices"] += result.get("valid_slices", 0)
                if result["overall_positive"]:
                    counters["n_positive"] += 1

                if verbose:
                    flag = "POSITIVE" if result["overall_positive"] else "negative"
                    print(flag)

            inference_queue.clear()

    elapsed = time.perf_counter() - t0
    throughput = (counters["total_slices"] / elapsed) if elapsed > 0 else 0

    counters["elapsed_seconds"]   = round(elapsed, 1)
    counters["slices_per_second"] = round(throughput, 1)

    if verbose:
        print(f"\n{'─'*68}")
        total_seen = sum(
            counters[k] for k in (
                "n_not_head_ct", "n_date_filtered", "n_already_processed",
                "n_no_ncct_series", "n_processed", "n_error"
            )
        )
        print(f"  Folders scanned      : {total_seen:>10,}")
        print(f"  ├─ Not head CT       : {counters['n_not_head_ct']:>10,}  "
              f"(pre-filter, stage 1)")
        print(f"  ├─ Date filtered     : {counters['n_date_filtered']:>10,}  "
              f"(DICOM StudyDate)")
        print(f"  ├─ Already in DB     : {counters['n_already_processed']:>10,}")
        print(f"  ├─ No NCCT series    : {counters['n_no_ncct_series']:>10,}  "
              f"(stage 2 selector)")
        print(f"  ├─ Errors            : {counters['n_error']:>10,}")
        print(f"  └─ Processed (infer) : {counters['n_processed']:>10,}")
        if counters["n_processed"] > 0:
            prev = counters["n_positive"] / counters["n_processed"]
            print(f"       Positive        : {counters['n_positive']:>10,}  "
                  f"({prev*100:.2f}% AI prevalence)")
            if not dry_run:
                print(f"       Slices inferred : {counters['total_slices']:>10,}  "
                      f"({throughput:.0f} slices/s)")
        h, rem = divmod(elapsed, 3600)
        m, s   = divmod(rem, 60)
        print(f"  Total elapsed        : {int(h):02d}:{int(m):02d}:{s:04.1f}")
        print(f"{'─'*68}\n")

    return counters


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scan a DICOM CT archive to build a local ICH prevalence database.\n"
            "Non-head CT studies are rejected cheaply before inference is run."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("archive_root", nargs="?", default=None,
        help="Root directory containing DICOM study folders")
    parser.add_argument("--days", type=int, default=None,
        help="Only include studies with DICOM StudyDate in the last N days")
    parser.add_argument("--checkpoint",
        default="./checkpoints_maxvit/best_maxvit_ich.pth")
    parser.add_argument("--db",
        default="./checkpoints_maxvit/prevalence.db")
    parser.add_argument("--progress-file", default=None,
        help="Path to scan-progress file (default: next to DB)")
    parser.add_argument("--no-skip", action="store_true",
        help="Reprocess studies already in the database")
    parser.add_argument("--dry-run", action="store_true",
        help="Run pre-filter and series selection without inference")
    parser.add_argument("--confirm", action="store_true",
        help="Required to run inference. Without this flag the scanner "
             "performs a dry-run only and prints what would be processed.")
    parser.add_argument("--io-workers", type=int, default=8,
        help="Parallel threads for DICOM header reading (default 8)")
    parser.add_argument("--report", action="store_true",
        help="Print prevalence summary after scanning")
    parser.add_argument("--report-only", action="store_true",
        help="Print reports without scanning")
    parser.add_argument("--trend", choices=["week", "month", "year"], default=None)
    parser.add_argument("--trend-location", default=None)
    parser.add_argument("--trend-periods", type=int, default=12)
    parser.add_argument("--min-n", type=int, default=10,
        help="Minimum studies required to display a prevalence estimate")
    args = parser.parse_args()

    db = PrevalenceDB(args.db)
    progress_file = Path(args.progress_file) if args.progress_file else None

    if not args.report_only:
        if args.archive_root is None:
            parser.error("archive_root is required unless --report-only is set")

        # ── Confirmation gate ─────────────────────────────────────────────────
        # Without --confirm, always run as dry-run and show what would happen.
        # This prevents accidental multi-hour GPU runs on a full archive.
        if not args.confirm and not args.dry_run:
            print()
            print("=" * 68)
            print("  ARCHIVE SCAN — DRY RUN (inference not started)")
            print("=" * 68)
            print(f"  Archive : {args.archive_root}")
            print(f"  Days    : {args.days or 'ALL (no date limit)'}")
            print()
            print("  --confirm was not supplied.")
            print("  Running dry-run to show scope before you commit.")
            print("=" * 68)
            print()
            scan_archive(
                root_dir        = args.archive_root,
                db              = db,
                checkpoint_path = args.checkpoint,
                days_back       = args.days,
                skip_existing   = not args.no_skip,
                dry_run         = True,          # forced
                io_workers      = args.io_workers,
                progress_file   = progress_file,
            )
            print("=" * 68)
            print("  Dry run complete.  To run inference on the studies above:")
            print()
            confirm_cmd = "  python prevalence_scanner.py " + args.archive_root
            if args.days:
                confirm_cmd += f" --days {args.days}"
            if args.no_skip:
                confirm_cmd += " --no-skip"
            confirm_cmd += " --confirm"
            print(confirm_cmd)
            print()
            print("  Add --report to print the prevalence summary when done.")
            print("=" * 68)
            print()
            sys.exit(0)

        # Confirmed (or explicit --dry-run) — proceed
        scan_archive(
            root_dir        = args.archive_root,
            db              = db,
            checkpoint_path = args.checkpoint,
            days_back       = args.days,
            skip_existing   = not args.no_skip,
            dry_run         = args.dry_run,
            io_workers      = args.io_workers,
            progress_file   = progress_file,
        )

    if args.report or args.report_only:
        db.print_summary(days=args.days, min_n=args.min_n)

    if args.trend:
        db.print_trend(
            location   = args.trend_location,
            period     = args.trend,
            n_periods  = args.trend_periods,
            min_n      = args.min_n,
        )


if __name__ == "__main__":
    main()
