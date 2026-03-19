#!/usr/bin/env python3
"""
Build demo study folders from the RSNA ICH training DICOM files.

Groups individual slices by StudyInstanceUID, selects studies that best
represent each ICH subtype (5 positive) plus 5 clean negatives, and
copies them into NewICH/demo_studies/ in a proper folder hierarchy:

    demo_studies/
        positive/
            subdural__ID_xxxxxxx/       ← StudyInstanceUID
                ID_xxxxxxx.dcm
                ID_xxxxxxx.dcm  ...
        negative/
            neg_01__ID_xxxxxxx/
                ...

Usage:
    python build_demo_studies.py
    python build_demo_studies.py --out demo_studies --n-pos 5 --n-neg 5
"""

import os
import sys
import csv
import json
import shutil
import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import pydicom

DICOM_DIR  = Path("/home/justinolddog/PyCharmMiscProject/stage_2_train")
LABELS_CSV = Path("/home/justinolddog/FullICH/RSNAICH/stage_2_train.csv")
OUT_DIR    = Path("/home/justinolddog/NewICH/demo_studies")

LABEL_COLS = ["epidural", "intraparenchymal", "intraventricular",
              "subarachnoid", "subdural", "any"]

# Preferred one study per subtype for maximum demo variety
PREFERRED_SUBTYPES = ["subdural", "epidural", "intraparenchymal",
                      "intraventricular", "subarachnoid"]


def load_labels(csv_path: Path) -> dict[str, dict[str, int]]:
    """
    Returns {image_id: {class: label, ...}, ...}
    Rows: ID_xxxx_epidural,0  →  image_id='ID_xxxx', class='epidural', label=0
    """
    print(f"Loading labels from {csv_path}...")
    labels: dict[str, dict[str, int]] = {}
    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)   # skip header
        for row in reader:
            if not row:
                continue
            parts    = row[0].rsplit("_", 1)
            img_id   = parts[0]
            cls      = parts[1] if len(parts) == 2 else "unknown"
            label    = int(row[1])
            if img_id not in labels:
                labels[img_id] = {}
            labels[img_id][cls] = label
    print(f"  {len(labels):,} labelled slices")
    return labels


def group_by_study(dicom_dir: Path,
                   labels: dict[str, dict[str, int]],
                   ) -> dict[str, dict]:
    """
    Walk dicom_dir, read StudyInstanceUID from each header, group slices.
    Returns {study_uid: {
        "study_uid": str,
        "files": [Path, ...],
        "slice_labels": {img_id: {class: label}},
        "study_positive": bool,
        "dominant_class": str,
        "any_label": int,
        "subtype_labels": {class: max_label_across_slices}
    }}
    """
    print(f"\nGrouping {len(list(dicom_dir.glob('*.dcm'))):,} DICOM files "
          f"by StudyInstanceUID...")

    studies: dict[str, dict] = {}

    dcm_files = sorted(dicom_dir.glob("*.dcm"))
    total = len(dcm_files)

    for i, dcm_file in enumerate(dcm_files):
        if (i + 1) % 20000 == 0:
            print(f"  {i+1:>7,} / {total:,}", flush=True)

        img_id = dcm_file.stem   # e.g. ID_000012eaf

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
        except Exception:
            continue

        if not study_uid:
            continue

        if study_uid not in studies:
            studies[study_uid] = {
                "study_uid":    study_uid,
                "files":        [],
                "slice_labels": {},
            }

        studies[study_uid]["files"].append(dcm_file)

        if img_id in labels:
            studies[study_uid]["slice_labels"][img_id] = labels[img_id]

    print(f"  {len(studies):,} unique studies found")

    # Aggregate study-level labels (positive if ANY slice is positive)
    for study in studies.values():
        agg = defaultdict(int)
        for sl in study["slice_labels"].values():
            for cls, lbl in sl.items():
                agg[cls] = max(agg[cls], lbl)

        study["subtype_labels"] = dict(agg)
        study["any_label"]      = agg.get("any", 0)
        study["study_positive"] = study["any_label"] == 1

        # Dominant class = first positive non-"any" subtype by priority
        dominant = ""
        for cls in PREFERRED_SUBTYPES:
            if agg.get(cls, 0) == 1:
                dominant = cls
                break
        study["dominant_class"] = dominant

    return studies


def select_studies(studies: dict[str, dict],
                   n_pos: int, n_neg: int,
                   min_slices: int = 20,
                   ) -> tuple[list[dict], list[dict]]:
    """
    Select n_pos positive studies (one per subtype if possible) and
    n_neg negative studies (well-labelled, sufficient slices).

    Preference: studies with more labelled slices (better coverage).
    """
    positives = [s for s in studies.values()
                 if s["study_positive"]
                 and len(s["slice_labels"]) >= min_slices]
    negatives = [s for s in studies.values()
                 if not s["study_positive"]
                 and len(s["slice_labels"]) >= min_slices
                 and s["subtype_labels"].get("any", 0) == 0]

    print(f"\n  Positive studies (≥{min_slices} labelled slices): "
          f"{len(positives):,}")
    print(f"  Negative studies (≥{min_slices} labelled slices): "
          f"{len(negatives):,}")

    # ── Positives: one per subtype, then fill remaining ───────────────────────
    selected_pos: list[dict] = []
    used_uids: set[str] = set()

    # One representative per subtype
    for subtype in PREFERRED_SUBTYPES:
        if len(selected_pos) >= n_pos:
            break
        candidates = sorted(
            [s for s in positives
             if s["dominant_class"] == subtype
             and s["study_uid"] not in used_uids],
            key=lambda s: -len(s["slice_labels"]),   # most labelled slices first
        )
        if candidates:
            chosen = candidates[0]
            selected_pos.append(chosen)
            used_uids.add(chosen["study_uid"])

    # Fill remaining slots if we didn't get 5 distinct subtypes
    for s in sorted(positives, key=lambda x: -len(x["slice_labels"])):
        if len(selected_pos) >= n_pos:
            break
        if s["study_uid"] not in used_uids:
            selected_pos.append(s)
            used_uids.add(s["study_uid"])

    # ── Negatives: highest labelled-slice count ────────────────────────────────
    selected_neg = sorted(negatives, key=lambda s: -len(s["slice_labels"]))[:n_neg]

    return selected_pos, selected_neg


def copy_study(study: dict, dest_dir: Path, label: str) -> Path:
    """
    Copy all DICOM slices for a study into dest_dir/<label__study_uid>/.
    Returns the destination folder.
    """
    cls     = study["dominant_class"] or label
    folder  = dest_dir / f"{cls}__{study['study_uid']}"
    folder.mkdir(parents=True, exist_ok=True)

    for src in study["files"]:
        shutil.copy2(src, folder / src.name)

    return folder


def write_manifest(selected_pos: list[dict],
                   selected_neg: list[dict],
                   out_dir: Path):
    """Write a JSON manifest describing the demo studies."""
    manifest = {
        "positive": [
            {
                "study_uid":     s["study_uid"],
                "dominant_class": s["dominant_class"],
                "subtype_labels": s["subtype_labels"],
                "n_slices":      len(s["files"]),
                "n_labelled":    len(s["slice_labels"]),
                "folder":        f"positive/{s['dominant_class']}__{s['study_uid']}",
            }
            for s in selected_pos
        ],
        "negative": [
            {
                "study_uid":  s["study_uid"],
                "n_slices":   len(s["files"]),
                "n_labelled": len(s["slice_labels"]),
                "folder":     f"negative/negative__{s['study_uid']}",
            }
            for s in selected_neg
        ],
    }
    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest → {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Build RSNA ICH demo study folders"
    )
    parser.add_argument("--dicom-dir",  default=str(DICOM_DIR))
    parser.add_argument("--labels-csv", default=str(LABELS_CSV))
    parser.add_argument("--out",        default=str(OUT_DIR))
    parser.add_argument("--n-pos",      type=int, default=5)
    parser.add_argument("--n-neg",      type=int, default=5)
    parser.add_argument("--min-slices", type=int, default=20,
        help="Minimum labelled slices per study (default 20)")
    args = parser.parse_args()

    dicom_dir  = Path(args.dicom_dir)
    labels_csv = Path(args.labels_csv)
    out_dir    = Path(args.out)

    if not dicom_dir.exists():
        sys.exit(f"DICOM directory not found: {dicom_dir}")
    if not labels_csv.exists():
        sys.exit(f"Labels CSV not found: {labels_csv}")

    print(f"\n{'='*64}")
    print(f"RSNA ICH Demo Study Builder")
    print(f"  DICOM source : {dicom_dir}")
    print(f"  Labels CSV   : {labels_csv}")
    print(f"  Output       : {out_dir}")
    print(f"  Positives    : {args.n_pos}  (one per subtype where possible)")
    print(f"  Negatives    : {args.n_neg}")
    print(f"{'='*64}\n")

    # ── Load labels ────────────────────────────────────────────────────────────
    labels = load_labels(labels_csv)

    # ── Group slices into studies ───────────────────────────────────────────────
    studies = group_by_study(dicom_dir, labels)

    # ── Select studies ─────────────────────────────────────────────────────────
    selected_pos, selected_neg = select_studies(
        studies, args.n_pos, args.n_neg, args.min_slices
    )

    print(f"\nSelected positives ({len(selected_pos)}):")
    for s in selected_pos:
        print(f"  {s['study_uid']:30s}  {s['dominant_class']:<22}  "
              f"{len(s['files']):3d} slices  {len(s['slice_labels']):3d} labelled")

    print(f"\nSelected negatives ({len(selected_neg)}):")
    for s in selected_neg:
        print(f"  {s['study_uid']:30s}  {'(negative)':<22}  "
              f"{len(s['files']):3d} slices  {len(s['slice_labels']):3d} labelled")

    # ── Copy files ─────────────────────────────────────────────────────────────
    print(f"\nCopying files to {out_dir}...")
    pos_dir = out_dir / "positive"
    neg_dir = out_dir / "negative"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    total_files = 0
    for s in selected_pos:
        dest = copy_study(s, pos_dir, "positive")
        n = len(list(dest.glob("*.dcm")))
        total_files += n
        print(f"  ✓ {dest.name}  ({n} files)")

    for s in selected_neg:
        s["dominant_class"] = "negative"
        dest = copy_study(s, neg_dir, "negative")
        n = len(list(dest.glob("*.dcm")))
        total_files += n
        print(f"  ✓ {dest.name}  ({n} files)")

    # ── Manifest ───────────────────────────────────────────────────────────────
    manifest = write_manifest(selected_pos, selected_neg, out_dir)

    print(f"\n{'='*64}")
    print(f"  Done.  {total_files:,} DICOM files copied to {out_dir}/")
    print(f"  Positive studies : {len(selected_pos)}")
    subtypes = [s['dominant_class'] for s in selected_pos]
    for st in subtypes:
        print(f"    • {st}")
    print(f"  Negative studies : {len(selected_neg)}")
    print(f"\n  Next step:")
    print(f"    python ich_agent.py {out_dir}/positive/<study_folder>")
    print(f"  Or process all at once:")
    print(f"    python run_demo_agent.py  (to be written)")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
