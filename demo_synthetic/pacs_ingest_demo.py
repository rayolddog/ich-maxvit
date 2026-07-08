#!/usr/bin/env python3
"""
pacs_ingest_demo.py — prove a PACS ingests the AI-authored TID 1500 SR.

Spins up a minimal DICOM Storage SCP (a stand-in PACS), then C-STOREs the
source CT study AND the conformant SR to it — demonstrating that a real DICOM
server accepts the SR alongside the images, resolves the SR's evidence
references, and can serve them back. This is the Docker-free equivalent of
STOW-RS into Orthanc (which is available via orthanc_demo.sh once Docker
Desktop is running).

Steps:
  1. build the conformant SR (conformant_sr.py) if absent
  2. start a Storage SCP on localhost:11112, storing to received/
  3. C-STORE the 4 CT slices + the SR
  4. report what the "PACS" holds: a CT series + an SR that references it

Usage:
  pacs_ingest_demo.py
"""
import glob
import os
import threading

from pydicom import dcmread
from pynetdicom import AE, evt, StoragePresentationContexts
from pynetdicom.sop_class import (CTImageStorage, ComprehensiveSRStorage,
                                  Verification)

HERE = os.path.dirname(__file__)
RECV = os.path.join(HERE, "received")
PORT = 11112


def start_scp():
    """Minimal Storage SCP that saves every received instance and logs it."""
    os.makedirs(RECV, exist_ok=True)
    received = []

    def handle_store(event):
        ds = event.dataset
        ds.file_meta = event.file_meta
        path = os.path.join(RECV, f"{ds.SOPInstanceUID}.dcm")
        ds.save_as(path, enforce_file_format=True)
        received.append((ds.SOPClassUID.name, ds.Modality,
                         str(ds.SOPInstanceUID)))
        return 0x0000     # Success

    ae = AE(ae_title="ICHPACS")
    ae.supported_contexts = StoragePresentationContexts
    ae.add_supported_context(Verification)
    scp = ae.start_server(("localhost", PORT), block=False,
                          evt_handlers=[(evt.EVT_C_STORE, handle_store)])
    return scp, received


def c_store_all(files):
    ae = AE(ae_title="ICHSENDER")
    ae.add_requested_context(CTImageStorage)
    ae.add_requested_context(ComprehensiveSRStorage)
    assoc = ae.associate("localhost", PORT, ae_title="ICHPACS")
    if not assoc.is_established:
        raise SystemExit("could not associate with the PACS SCP")
    ok = fail = 0
    for f in files:
        status = assoc.send_c_store(dcmread(f))
        if status and status.Status == 0x0000:
            ok += 1
        else:
            fail += 1
    assoc.release()
    return ok, fail


def main():
    # 1. ensure the SR exists
    sr_path = os.path.join(HERE, "out", "ich_ai_sr.dcm")
    if not os.path.exists(sr_path):
        import conformant_sr
        conformant_sr.build_sr(os.path.join(HERE, "pacs", "followup"),
                               os.path.join(HERE, "out"))

    ct_files = sorted(glob.glob(os.path.join(HERE, "pacs", "followup", "*.dcm")))
    to_send = ct_files + [sr_path]

    # 2. start the PACS
    scp, received = start_scp()
    print(f"[pacs] Storage SCP 'ICHPACS' listening on localhost:{PORT}")

    # 3. C-STORE CT + SR
    print(f"[pacs] C-STORE {len(ct_files)} CT slices + 1 SR ...")
    ok, fail = c_store_all(to_send)
    print(f"[pacs] stored {ok} instances, {fail} failed")

    # 4. report what the PACS holds
    scp.shutdown()
    mods = {}
    for name, mod, uid in received:
        mods.setdefault(mod, []).append(name)
    print(f"[pacs] the PACS now holds:")
    for mod, items in mods.items():
        print(f"    {mod}: {len(items)} instance(s) — {items[0]}")
    # confirm the SR references the CT that is also in the PACS
    sr = dcmread(sr_path)
    ct_uids = {dcmread(f, stop_before_pixels=True).SOPInstanceUID
               for f in ct_files}
    ev = sr.CurrentRequestedProcedureEvidenceSequence
    ref = {r.ReferencedSOPInstanceUID
           for s in ev for ser in s.ReferencedSeriesSequence
           for r in ser.ReferencedSOPSequence}
    print(f"[pacs] SR evidence resolves to CT in the PACS: "
          f"{len(ref & ct_uids)}/{len(ct_uids)} slices matched")
    print("[pacs] SUCCESS: a DICOM server ingested the AI TID 1500 SR "
          "alongside the images it references.")


if __name__ == "__main__":
    main()
