#!/usr/bin/env python3
"""
Run the ICH agent on all demo studies and populate the worklist.

Reads demo_studies/manifest.json, processes each study folder through
ich_agent.run_agent(), and records results to both the prevalence DB
and the worklist server (if running).

Usage:
    python run_demo_agent.py
    python run_demo_agent.py --demo-dir demo_studies
    python run_demo_agent.py --skip-existing   # skip study_uids already in worklist
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ich_agent import run_agent

DEMO_DIR     = Path(__file__).parent / "demo_studies"
MANIFEST     = DEMO_DIR / "manifest.json"
WORKLIST_JSON = Path(__file__).parent / "worklist.json"

# Clinical indications that match each study type for realistic context
INDICATION_MAP = {
    "subdural":         "Fall, altered mental status",
    "epidural":         "Head trauma, loss of consciousness",
    "intraparenchymal": "Sudden onset headache, hypertension",
    "intraventricular": "Decreased level of consciousness",
    "subarachnoid":     "Thunderclap headache, worst of life",
    "negative":         "Headache, rule out intracranial pathology",
}

# Simulated patient names for the demo (not real patients)
DEMO_NAMES = {
    "subdural":         "DEMO^SUBDURAL",
    "epidural":         "DEMO^EPIDURAL",
    "intraparenchymal": "DEMO^INTRAPARENCHYMAL",
    "intraventricular": "DEMO^INTRAVENTRICULAR",
    "subarachnoid":     "DEMO^SUBARACHNOID",
    "negative":         "DEMO^NEGATIVE",
}


def load_existing_uids() -> set[str]:
    """Return study_uids already recorded in worklist.json."""
    if not WORKLIST_JSON.exists():
        return set()
    try:
        with open(WORKLIST_JSON) as f:
            studies = json.load(f)
        return {s.get("study_uid", "") for s in studies}
    except Exception:
        return set()


def run_all(demo_dir: Path, skip_existing: bool, verbose: bool) -> dict:
    manifest_path = demo_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}\n"
                 f"Run build_demo_studies.py first.")

    with open(manifest_path) as f:
        manifest = json.load(f)

    existing_uids = load_existing_uids() if skip_existing else set()

    # Build ordered study list: positives first, then negatives
    studies = []
    for entry in manifest.get("positive", []):
        studies.append({**entry, "group": "positive"})
    for i, entry in enumerate(manifest.get("negative", [])):
        studies.append({**entry, "group": "negative",
                        "dominant_class": "negative",
                        "patient_suffix": str(i + 1)})

    counters = {"processed": 0, "skipped": 0, "errors": 0,
                "positive": 0, "negative": 0}
    t0 = time.perf_counter()

    print(f"\n{'='*64}")
    print(f"ICH Demo Agent Runner")
    print(f"  Studies : {len(studies)} ({len(manifest['positive'])} positive, "
          f"{len(manifest['negative'])} negative)")
    print(f"  Demo dir: {demo_dir}")
    print(f"{'='*64}\n")

    for i, entry in enumerate(studies):
        study_uid    = entry.get("study_uid", "")
        folder_rel   = entry.get("folder", "")
        dominant     = entry.get("dominant_class", "negative")
        group        = entry.get("group", "negative")
        folder       = demo_dir / folder_rel

        print(f"[{i+1}/{len(studies)}] {dominant.upper():<22}  {study_uid}")

        if not folder.exists():
            print(f"  SKIP — folder not found: {folder}\n")
            counters["skipped"] += 1
            continue

        if skip_existing and study_uid in existing_uids:
            print(f"  SKIP — already in worklist\n")
            counters["skipped"] += 1
            continue

        indication = INDICATION_MAP.get(dominant,
                                        "Head CT — rule out intracranial pathology")

        try:
            result = run_agent(
                study_folder = str(folder),
                indication   = indication,
                verbose      = verbose,
            )
            counters["processed"] += 1
            if group == "positive":
                counters["positive"] += 1
            else:
                counters["negative"] += 1

            # Print a short summary of what the agent returned
            lines = [l for l in result.splitlines() if l.strip()]
            if lines:
                print(f"\n  Agent summary:")
                for line in lines[:4]:
                    print(f"    {line.strip()}")
            print()

        except Exception as exc:
            print(f"  ERROR: {exc}\n")
            counters["errors"] += 1

    elapsed = time.perf_counter() - t0
    h, rem  = divmod(elapsed, 3600)
    m, s    = divmod(rem, 60)

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
        description="Run ICH agent on all demo studies"
    )
    parser.add_argument("--demo-dir", default=str(DEMO_DIR),
        help="Path to demo_studies directory")
    parser.add_argument("--skip-existing", action="store_true",
        help="Skip studies already recorded in worklist.json")
    parser.add_argument("--quiet", action="store_true",
        help="Suppress per-tool verbose output from the agent")
    args = parser.parse_args()

    run_all(
        demo_dir     = Path(args.demo_dir),
        skip_existing = args.skip_existing,
        verbose       = not args.quiet,
    )


if __name__ == "__main__":
    main()
