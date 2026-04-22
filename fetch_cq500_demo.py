#!/usr/bin/env python3
"""
Download a small subset of CQ500 head CT studies for demo use.

Fetches the qure.ai reads.csv labels, selects one study per ICH subtype
(subdural, epidural, intraparenchymal, intraventricular, subarachnoid) plus
n_neg negatives using majority vote across three readers, then downloads only
those studies from S3 and restructures them into the demo_studies/ format
expected by run_demo_agent.py.

Dataset : CQ500 (qure.ai), CC-BY-NC-SA 4.0
Citation: Chilamkurthy et al., Lancet 2018.  http://headctstudy.qure.ai

Usage:
    python fetch_cq500_demo.py
    python fetch_cq500_demo.py --out demo_studies --n-neg 3
    python fetch_cq500_demo.py --out demo_studies_cq500   # separate dir
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

import pydicom
import requests

S3_BASE       = "https://s3.ap-south-1.amazonaws.com/qure.headct.study"
READS_CSV_URL = f"{S3_BASE}/reads.csv"

# CQ500 column short codes for each ICH subtype
SUBTYPE_CODES: dict[str, str] = {
    "subdural":         "SDH",
    "epidural":         "EDH",
    "intraparenchymal": "IPH",
    "intraventricular": "IVH",
    "subarachnoid":     "SAH",
}
READERS = ["R1", "R2", "R3"]


# ── Label helpers ──────────────────────────────────────────────────────────────

def majority(row: dict, code: str) -> bool:
    """True if at least 2 of 3 readers flagged this finding."""
    return sum(int(row.get(f"{r}:{code}", 0) or 0) for r in READERS) >= 2


def subtype_labels_from_row(row: dict) -> dict[str, int]:
    """Build the subtype_labels dict matching the manifest schema."""
    labels: dict[str, int] = {}
    any_positive = 0
    for subtype, code in SUBTYPE_CODES.items():
        v = 1 if majority(row, code) else 0
        labels[subtype] = v
        any_positive = max(any_positive, v)
    labels["any"] = any_positive
    return labels


# ── Study selection ────────────────────────────────────────────────────────────

def download_reads_csv() -> list[dict]:
    print(f"Downloading reads.csv ...", end=" ", flush=True)
    r = requests.get(READS_CSV_URL, timeout=30)
    r.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(r.text)))
    print(f"{len(rows)} studies")
    return rows


def select_studies(
    rows: list[dict], n_neg: int
) -> tuple[list[tuple[str, dict]], list[dict]]:
    """
    Returns:
        positives : [(subtype, row), ...] — one per subtype, majority-vote positive
        negatives : [row, ...]            — all three readers agree: no ICH
    """
    positives: dict[str, dict] = {}
    used_names: set[str] = set()   # prevent same study appearing under two subtypes
    negatives: list[dict] = []
    neg_names: set[str] = set()

    for row in rows:
        name = row["name"]

        # Negative: all three readers say no ICH, not already selected
        if len(negatives) < n_neg and name not in neg_names:
            if all(int(row.get(f"{r}:ICH", 0) or 0) == 0 for r in READERS):
                negatives.append(row)
                neg_names.add(name)

        # One positive per subtype — skip if this study is already used
        if name not in used_names:
            for subtype, code in SUBTYPE_CODES.items():
                if subtype not in positives and majority(row, code):
                    positives[subtype] = row
                    used_names.add(name)
                    break   # assign each study to at most one subtype

        if len(positives) == len(SUBTYPE_CODES) and len(negatives) >= n_neg:
            break

    return list(positives.items()), negatives


# ── Download & extraction ──────────────────────────────────────────────────────

def study_number(name: str) -> int:
    """'CQ500-CT-42' → 42"""
    return int(name.rsplit("-", 1)[-1])


def get_zip(n: int, cache_dir: Path) -> bytes:
    """Return zip bytes from cache if present, otherwise download and cache."""
    cache_path = cache_dir / f"CQ500-CT-{n}.zip"
    if cache_path.exists() and cache_path.stat().st_size > 0:
        print(f"  CQ500-CT-{n}.zip  (cached, {cache_path.stat().st_size / 1_048_576:.1f} MB)")
        return cache_path.read_bytes()

    url = f"{S3_BASE}/CQ500-CT-{n}.zip"
    print(f"  Downloading CQ500-CT-{n}.zip ...", end=" ", flush=True)
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    data = r.content
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    print(f"{len(data) / 1_048_576:.1f} MB  (saved to cache)")
    return data


def best_series_members(zf: zipfile.ZipFile) -> list[str]:
    """
    Return the zip member paths for the series with the most .dcm files.
    Prefer series whose folder name contains 'THIN' (thinnest slices).
    """
    groups: dict[str, list[str]] = {}
    for name in zf.namelist():
        if name.lower().endswith(".dcm"):
            parent = "/".join(name.split("/")[:-1])
            groups.setdefault(parent, []).append(name)

    if not groups:
        return []

    def score(item: tuple[str, list[str]]) -> tuple[int, int]:
        path, files = item
        return (1 if "thin" in path.lower() else 0, len(files))

    _, members = max(groups.items(), key=score)
    return members


def extract_to(zip_bytes: bytes, dest: Path, max_slices: int = 30) -> tuple[int, str]:
    """
    Extract best series into dest/, subsampling to at most max_slices evenly
    spaced slices.  Returns (n_slices, study_instance_uid).
    """
    import math
    dest.mkdir(parents=True, exist_ok=True)
    study_uid = ""

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = sorted(best_series_members(zf))
        if not members:
            raise ValueError("No DICOM files found in zip")

        # Evenly subsample if the series is longer than max_slices
        if max_slices and len(members) > max_slices:
            step = len(members) / max_slices
            members = [members[int(i * step)] for i in range(max_slices)]

        for i, member in enumerate(members):
            data = zf.read(member)
            out_path = dest / f"CT{i:06d}.dcm"
            out_path.write_bytes(data)

            if not study_uid:
                try:
                    ds = pydicom.dcmread(str(out_path), stop_before_pixels=True)
                    study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
                except Exception:
                    pass

    return len(list(dest.glob("*.dcm"))), study_uid


# ── Manifest ───────────────────────────────────────────────────────────────────

def write_manifest(
    pos_entries: list[dict],
    neg_entries: list[dict],
    out_dir: Path,
) -> None:
    manifest = {"positive": pos_entries, "negative": neg_entries}
    p = out_dir / "manifest.json"
    with open(p, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest → {p}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch CQ500 demo studies (qure.ai, CC-BY-NC-SA 4.0)"
    )
    parser.add_argument("--out",       default="demo_studies",
                        help="Output directory (default: demo_studies)")
    parser.add_argument("--cache-dir", default="cq500_cache",
                        help="Directory for cached zip files (default: cq500_cache)")
    parser.add_argument("--n-neg",      type=int, default=2,
                        help="Negative studies to include (default: 2)")
    parser.add_argument("--max-slices", type=int, default=30,
                        help="Max slices per study, evenly subsampled (default: 30, 0=all)")
    args = parser.parse_args()

    out_dir    = Path(args.out)
    cache_dir  = Path(args.cache_dir)
    max_slices = args.max_slices or 0
    pos_dir   = out_dir / "positive"
    neg_dir   = out_dir / "negative"

    print(f"\n{'='*64}")
    print("CQ500 Demo Study Fetcher")
    print("Source : qure.ai Head CT Study, CC-BY-NC-SA 4.0")
    print(f"Output : {out_dir.resolve()}")
    print(f"Cache  : {cache_dir.resolve()}")
    print(f"{'='*64}\n")

    rows = download_reads_csv()
    positives, negatives = select_studies(rows, args.n_neg)

    print(f"\nSelected studies:")
    for subtype, row in positives:
        print(f"  + {subtype:<22}  {row['name']}")
    for row in negatives:
        print(f"  - {'(negative)':<22}  {row['name']}")

    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    pos_manifest: list[dict] = []
    neg_manifest: list[dict] = []

    print(f"\nDownloading positive studies...")
    for subtype, row in positives:
        n = study_number(row["name"])
        zip_bytes = get_zip(n, cache_dir)
        safe_name = row["name"].replace("-", "_")
        dest = pos_dir / f"{subtype}__{safe_name}"
        n_slices, study_uid = extract_to(zip_bytes, dest, max_slices)
        uid = study_uid or safe_name
        print(f"    {dest.name}  ({n_slices} slices)")
        pos_manifest.append({
            "study_uid":      uid,
            "dominant_class": subtype,
            "subtype_labels": subtype_labels_from_row(row),
            "n_slices":       n_slices,
            "n_labelled":     n_slices,
            "source":         "CQ500",
            "cq500_name":     row["name"],
            "folder":         f"positive/{dest.name}",
        })

    print(f"\nDownloading negative studies...")
    for row in negatives:
        n = study_number(row["name"])
        zip_bytes = get_zip(n, cache_dir)
        safe_name = row["name"].replace("-", "_")
        dest = neg_dir / f"negative__{safe_name}"
        n_slices, study_uid = extract_to(zip_bytes, dest, max_slices)
        uid = study_uid or safe_name
        print(f"    {dest.name}  ({n_slices} slices)")
        neg_manifest.append({
            "study_uid":  uid,
            "n_slices":   n_slices,
            "n_labelled": n_slices,
            "source":     "CQ500",
            "cq500_name": row["name"],
            "folder":     f"negative/{dest.name}",
        })

    write_manifest(pos_manifest, neg_manifest, out_dir)

    total = sum(e["n_slices"] for e in pos_manifest + neg_manifest)
    n_studies = len(pos_manifest) + len(neg_manifest)
    print(f"\n{'='*64}")
    print(f"Done.  {total:,} slices across {n_studies} studies in {out_dir}/")
    print(f"\nAttribution: qure.ai Head CT Study (CC-BY-NC-SA 4.0)")
    print(f"  http://headctstudy.qure.ai/dataset")
    print(f"\nNext:")
    print(f"  python ich_worklist.py         # Terminal 1 — start worklist server")
    print(f"  python run_demo_agent.py --demo-dir {out_dir}  # Terminal 2")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
