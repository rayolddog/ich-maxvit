#!/usr/bin/env python3
"""
orthanc_stow.py — STOW-RS the CT study + AI SR into a running Orthanc, then
QIDO-verify. The DICOMweb (web-UI) version of pacs_ingest_demo.py.

Prereq: Orthanc running with DICOMweb at http://localhost:8042 (see
orthanc_demo.sh). Requires Docker Desktop to be running.
"""
import glob
import os

from dicomweb_client.api import DICOMwebClient
from pydicom import dcmread

HERE = os.path.dirname(__file__)
BASE = "http://localhost:8042/dicom-web"


def main():
    sr_path = os.path.join(HERE, "out", "ich_ai_sr.dcm")
    ct = sorted(glob.glob(os.path.join(HERE, "pacs", "followup", "*.dcm")))
    datasets = [dcmread(f) for f in ct] + [dcmread(sr_path)]

    client = DICOMwebClient(url=BASE)
    client.store_instances(datasets=datasets)                 # STOW-RS
    print(f"[stow] STOW-RS sent {len(datasets)} instances (4 CT + 1 SR)")

    studies = client.search_for_studies()                     # QIDO-RS
    print(f"[stow] Orthanc now indexes {len(studies)} study(ies):")
    for s in studies:
        uid = s.get("0020000D", {}).get("Value", ["?"])[0]
        mods = s.get("00080061", {}).get("Value", [])
        print(f"    study {uid}  modalities {mods}")
    print("[stow] Open the Orthanc UI at http://localhost:8042/ui/app/ "
          "to see the SR alongside the CT.")


if __name__ == "__main__":
    main()
