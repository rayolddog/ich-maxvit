#!/usr/bin/env python3
"""
Run the ICH pipeline on all demo studies WITHOUT the Claude API.

Calls the tool implementations in ich_agent directly (scan → series select →
inference → DICOM SR → worklist) and generates templated report text.

Usage:
    python run_demo_direct.py
    python run_demo_direct.py --demo-dir demo_studies
    python run_demo_direct.py --skip-existing
"""

import os
import sys
import json
import time
import warnings
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress RSNA invalid-UID warnings
warnings.filterwarnings("ignore", category=UserWarning)

from ich_agent import (
    _scan_study,
    _run_ich_inference,
    _generate_dicom_sr,
    _flag_worklist,
    CHECKPOINT_PATH,
)

try:
    from ich_worklist import record_result as _worklist_record
    _WORKLIST_AVAILABLE = True
except ImportError:
    _WORKLIST_AVAILABLE = False
    print("[WARN] ich_worklist not available — worklist will not be updated")

try:
    from prevalence_db import PrevalenceDB, normalize_location
    _PREVALENCE_DB = PrevalenceDB()
    _PREVALENCE_DB_AVAILABLE = True
except ImportError:
    _PREVALENCE_DB_AVAILABLE = False
    print("[WARN] prevalence_db not available")

DEMO_DIR      = Path(__file__).parent / "demo_studies"
MANIFEST      = DEMO_DIR / "manifest.json"
WORKLIST_JSON = Path(__file__).parent / "worklist.json"

# Prevalence defaults by indication keyword
PREVALENCE_MAP = {
    "trauma":          0.10,
    "loss of consciousness": 0.10,
    "altered mental":  0.08,
    "hypertension":    0.06,
    "thunderclap":     0.15,
    "worst of life":   0.15,
    "headache":        0.02,
}

DEMO_NAMES = {
    "subdural":         "DEMO^SUBDURAL",
    "epidural":         "DEMO^EPIDURAL",
    "intraparenchymal": "DEMO^INTRAPARENCHYMAL",
    "intraventricular": "DEMO^INTRAVENTRICULAR",
    "subarachnoid":     "DEMO^SUBARACHNOID",
    "negative":         "DEMO^NEGATIVE",
}

INDICATION_MAP = {
    "subdural":         "Fall, altered mental status",
    "epidural":         "Head trauma, loss of consciousness",
    "intraparenchymal": "Sudden onset headache, hypertension",
    "intraventricular": "Decreased level of consciousness",
    "subarachnoid":     "Thunderclap headache, worst of life",
    "negative":         "Headache, rule out intracranial pathology",
}

SUBTYPE_LONG = {
    "subdural":         "subdural hematoma",
    "epidural":         "epidural hematoma",
    "intraparenchymal": "intraparenchymal hemorrhage",
    "intraventricular": "intraventricular hemorrhage",
    "subarachnoid":     "subarachnoid hemorrhage",
}


def _indication_prevalence(indication: str) -> float:
    ind_lower = indication.lower()
    for kw, prev in PREVALENCE_MAP.items():
        if kw in ind_lower:
            return prev
    return 0.05


def _select_series(scan_result: dict) -> dict | None:
    """Pick the best CT series — for demo studies there is always exactly one."""
    series_list = scan_result.get("series", [])
    ct_series = [s for s in series_list if s.get("modality") == "CT"]
    if not ct_series:
        return None
    # Prefer non-derived, prefer thinnest slice thickness
    def sort_key(s):
        img_type = s.get("image_type", [])
        is_derived = 1 if img_type and img_type[0] == "DERIVED" else 0
        try:
            thick = float(s.get("slice_thickness", "99") or "99")
        except ValueError:
            thick = 99.0
        return (is_derived, thick)
    return sorted(ct_series, key=sort_key)[0]


def _generate_report(inference: dict, indication: str, prevalence: float,
                     metrics: dict, slice_count: int) -> tuple[str, str]:
    """Generate a templated report paragraph and impression bullet."""
    positive    = inference.get("overall_positive", False)
    dom_class   = inference.get("dominant_class", "")
    study_level = inference.get("study_level", {})
    hot_slices  = inference.get("hot_slices", {})

    any_prob = study_level.get("any", {}).get("prob", 0.0)
    sens     = metrics.get("sensitivity", 0.94)
    spec     = metrics.get("specificity", 0.95)
    ppv      = metrics.get("ppv", 0.0)
    npv      = metrics.get("npv", 0.0)

    if positive:
        long_name  = SUBTYPE_LONG.get(dom_class, dom_class)
        dom_prob   = study_level.get(dom_class, {}).get("prob", any_prob)
        ppv_pct    = round(ppv * 100, 1)

        # Collect positive subtypes
        pos_subtypes = [
            SUBTYPE_LONG.get(cls, cls)
            for cls, v in study_level.items()
            if cls != "any" and v.get("positive")
        ]
        pos_str = ", ".join(pos_subtypes) if pos_subtypes else long_name

        # Hot slice range (hot_slices is a list of dicts with "slice_index")
        hot_ranges = []
        if isinstance(hot_slices, list):
            hot_ranges = [h["slice_index"] + 1 for h in hot_slices
                          if isinstance(h, dict) and "slice_index" in h]
        slice_note = ""
        if hot_ranges:
            mn, mx = min(hot_ranges), max(hot_ranges)
            slice_note = (f" Most prominent finding at slice {mn}"
                          + (f"–{mx}" if mx != mn else "") + ".")

        paragraph = (
            f"AI-assisted screening for intracranial hemorrhage (ICH) was performed "
            f"on the submitted axial noncontrast CT head series ({slice_count} slices) "
            f"using a MaxViT-based classifier (mean AUC 0.986, RSNA ICH dataset). "
            f"The study is POSITIVE for suspected {pos_str} (model probability "
            f"{dom_prob*100:.0f}%; Sens {sens*100:.0f}%, Spec {spec*100:.0f}%; "
            f"PPV {ppv_pct}% at {prevalence*100:.0f}% local prevalence).{slice_note} "
            f"This study has been flagged for priority radiologist review. "
            f"AI-assisted screening. Not FDA-cleared. Requires radiologist confirmation."
        )
        bullet = (
            f"• Suspected {long_name} identified by AI screening "
            f"(model score {dom_prob*100:.0f}%, PPV {ppv_pct}% at {prevalence*100:.0f}% prevalence). "
            f"Study flagged for priority AI-detected review. "
            f"AI-assisted screening. Not FDA-cleared. Requires radiologist confirmation."
        )
    else:
        npv_pct = round(npv * 100, 1)
        paragraph = (
            f"AI-assisted screening for intracranial hemorrhage (ICH) was performed "
            f"on the submitted axial noncontrast CT head series ({slice_count} slices) "
            f"using a MaxViT-based classifier (mean AUC 0.986, RSNA ICH dataset). "
            f"The study is NEGATIVE for ICH (model probability {any_prob*100:.0f}%; "
            f"Sens {sens*100:.0f}%, Spec {spec*100:.0f}%; "
            f"NPV {npv_pct}% at {prevalence*100:.0f}% local prevalence). "
            f"Routine radiologist read is appropriate. "
            f"AI-assisted screening. Not FDA-cleared. Requires radiologist confirmation."
        )
        bullet = (
            f"• No ICH detected by AI screening "
            f"(NPV {npv_pct}% at {prevalence*100:.0f}% prevalence). "
            f"Routine read appropriate. "
            f"AI-assisted screening. Not FDA-cleared. Requires radiologist confirmation."
        )

    return paragraph, bullet


def load_existing_uids() -> set[str]:
    if not WORKLIST_JSON.exists():
        return set()
    try:
        with open(WORKLIST_JSON) as f:
            studies = json.load(f)
        return {s.get("study_uid", "") for s in studies}
    except Exception:
        return set()


def process_study(entry: dict, demo_dir: Path, verbose: bool) -> str:
    """Process a single study entry from the manifest. Returns a status string."""
    study_uid   = entry.get("study_uid", "")
    folder_rel  = entry.get("folder", "")
    dominant    = entry.get("dominant_class", "negative")
    folder      = demo_dir / folder_rel
    indication  = INDICATION_MAP.get(dominant, "Head CT — rule out intracranial pathology")
    patient_name = DEMO_NAMES.get(dominant, f"DEMO^{dominant.upper()}")

    if not folder.exists():
        return f"SKIP — folder not found: {folder}"

    # ── 1. Scan study ──────────────────────────────────────────────────────────
    if verbose:
        print("  [1/5] Scanning DICOM headers...", flush=True)
    scan = _scan_study(str(folder))
    if "error" in scan:
        return f"ERROR in scan_study: {scan['error']}"

    # ── 2. Select series ───────────────────────────────────────────────────────
    series = _select_series(scan)
    if series is None:
        return "ERROR — no CT series found"
    series_folder = series["series_folder"]
    slice_count   = series["slice_count"]
    if verbose:
        print(f"  [2/5] Selected series: {series['series_description'] or '(unnamed)'} "
              f"— {slice_count} slices, folder: {Path(series_folder).name}", flush=True)

    # ── 3. Look up local prevalence ────────────────────────────────────────────
    prevalence = _indication_prevalence(indication)
    if _PREVALENCE_DB_AVAILABLE:
        local_prev, _ = _PREVALENCE_DB.best_prevalence_for_agent(
            location=None, days=365, min_n=30)
        if local_prev is not None:
            prevalence = local_prev

    # ── 4. Run inference ───────────────────────────────────────────────────────
    if verbose:
        print("  [3/5] Running MaxViT inference...", flush=True)
    inference = _run_ich_inference(series_folder, CHECKPOINT_PATH)
    if "error" in inference:
        return f"ERROR in inference: {inference['error']}"

    ai_positive   = inference.get("overall_positive", False)
    dom_class_raw = inference.get("dominant_class", "")

    if verbose:
        sl = inference.get("study_level", {})
        print(f"  Inference result: {'POSITIVE' if ai_positive else 'negative'} "
              f"(any prob={sl.get('any',{}).get('prob',0)*100:.0f}%, "
              f"dominant={dom_class_raw or 'none'})", flush=True)

    # ── 5. Generate DICOM SR ───────────────────────────────────────────────────
    study_metadata = {
        "study_uid":  study_uid,
        "patient_id": study_uid,
        "indication": indication,
    }
    output_path = str(folder / "ich_ai_sr.json")
    if verbose:
        print("  [4/5] Generating DICOM SR...", flush=True)
    sr_result = _generate_dicom_sr(inference, study_metadata, output_path, prevalence)
    sr_path  = sr_result.get("output_path", "")
    metrics  = sr_result.get("metrics", {})

    # ── 6. Generate report text ────────────────────────────────────────────────
    paragraph, bullet = _generate_report(
        inference, indication, prevalence, metrics, slice_count)

    # ── 7. Flag worklist (print + record) ─────────────────────────────────────
    _flag_worklist(study_uid, ai_positive, dom_class_raw)

    # ── 8. Record in prevalence DB ─────────────────────────────────────────────
    if _PREVALENCE_DB_AVAILABLE:
        try:
            _PREVALENCE_DB.record_study(
                study_uid         = study_uid,
                ai_positive       = ai_positive,
                patient_id        = study_uid,
                dominant_class    = dom_class_raw,
                study_level_probs = inference.get("study_level", {}),
                series_folder     = series_folder,
            )
            if verbose:
                print(f"  [PREV DB] Study recorded", flush=True)
        except Exception as exc:
            if verbose:
                print(f"  [PREV DB] Warning: {exc}", flush=True)

    # ── 9. Record in worklist ─────────────────────────────────────────────────
    if verbose:
        print("  [5/5] Recording in worklist...", flush=True)
    if _WORKLIST_AVAILABLE:
        try:
            _worklist_record(
                study_uid         = study_uid,
                patient_id        = study_uid,
                indication        = indication,
                ai_positive       = ai_positive,
                ai_result         = inference,
                report_paragraph  = paragraph,
                impression_bullet = bullet,
                sr_path           = sr_path,
                metrics           = metrics,
                prevalence        = prevalence,
                description       = "CT HEAD W/O CONTRAST",
                patient_name      = patient_name,
            )
        except Exception as exc:
            return f"ERROR in record_result: {exc}"

    status = "POSITIVE" if ai_positive else "negative"
    return f"OK — {status}"


def run_all(demo_dir: Path, skip_existing: bool, verbose: bool) -> dict:
    manifest_path = demo_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}\n"
                 f"Run build_demo_studies.py first.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    existing_uids = load_existing_uids() if skip_existing else set()

    studies = []
    for entry in manifest.get("positive", []):
        studies.append({**entry, "group": "positive"})
    for entry in manifest.get("negative", []):
        studies.append({**entry, "group": "negative",
                        "dominant_class": "negative"})

    counters = {"processed": 0, "skipped": 0, "errors": 0,
                "positive": 0, "negative": 0}
    t0 = time.perf_counter()

    print(f"\n{'='*64}")
    print(f"ICH Demo Direct Runner  (no Claude API required)")
    print(f"  Studies : {len(studies)} ({len(manifest['positive'])} positive, "
          f"{len(manifest['negative'])} negative)")
    print(f"  Demo dir: {demo_dir}")
    print(f"{'='*64}\n")

    for i, entry in enumerate(studies):
        study_uid = entry.get("study_uid", "")
        dominant  = entry.get("dominant_class", "negative")
        group     = entry.get("group", "negative")

        print(f"[{i+1}/{len(studies)}] {dominant.upper():<22}  {study_uid}")

        if skip_existing and study_uid in existing_uids:
            print(f"  SKIP — already in worklist\n")
            counters["skipped"] += 1
            continue

        t1 = time.perf_counter()
        try:
            status = process_study(entry, demo_dir, verbose)
        except Exception as exc:
            status = f"ERROR: {exc}"

        elapsed = time.perf_counter() - t1
        print(f"  → {status}  ({elapsed:.1f}s)\n")

        if status.startswith("ERROR") or status.startswith("SKIP"):
            counters["errors"] += 1
        else:
            counters["processed"] += 1
            if group == "positive":
                counters["positive"] += 1
            else:
                counters["negative"] += 1

    elapsed = time.perf_counter() - t0
    h, rem = divmod(elapsed, 3600)
    m, s   = divmod(rem, 60)

    print(f"\n{'='*64}")
    print(f"  Processed : {counters['processed']}  "
          f"({counters['positive']} positive, {counters['negative']} negative)")
    print(f"  Skipped   : {counters['skipped']}")
    print(f"  Errors    : {counters['errors']}")
    print(f"  Elapsed   : {int(h):02d}:{int(m):02d}:{s:04.1f}")
    print(f"{'='*64}\n")

    return counters


def main():
    parser = argparse.ArgumentParser(
        description="Run ICH pipeline on demo studies (no Claude API)"
    )
    parser.add_argument("--demo-dir", default=str(DEMO_DIR),
        help="Path to demo_studies directory")
    parser.add_argument("--skip-existing", action="store_true",
        help="Skip studies already in worklist.json")
    parser.add_argument("--quiet", action="store_true",
        help="Suppress per-step verbose output")
    args = parser.parse_args()

    run_all(
        demo_dir      = Path(args.demo_dir),
        skip_existing = args.skip_existing,
        verbose       = not args.quiet,
    )


if __name__ == "__main__":
    main()
