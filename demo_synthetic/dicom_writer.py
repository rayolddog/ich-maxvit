#!/usr/bin/env python3
"""
dicom_writer.py — write a phantom HU volume as a linked CT DICOM series.

Deterministic UIDs and dates so the fictional EMR/PACS can reference the two
studies by StudyInstanceUID. Single source of truth for the demo case's
identifiers: STUDY_META below (imported by emr_pacs.py).

All identifiers are fictional. Patient "DEMO^SUBDURAL", MRN DEMO-0001.
The two studies are a short-interval pair on one simulated 2024-01-15 day.
"""
import os

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

FOV_MM = 230.0
N = 512
PX = FOV_MM / N
SLICE_MM = 5.0
ROOT = "1.2.826.0.1.3680043.10.9999"     # fictional demo org root

STUDY_META = {
    "baseline": dict(
        study_uid=f"{ROOT}.1.1", series_uid=f"{ROOT}.1.2",
        accession="DEMOACC001", study_date="20240115", study_time="083000",
        description="CT HEAD WO CONTRAST",
        indication="Ground-level fall, head strike; on apixaban",
        series_desc="AXIAL 5MM",
    ),
    "followup": dict(
        study_uid=f"{ROOT}.2.1", series_uid=f"{ROOT}.2.2",
        accession="DEMOACC002", study_date="20240115", study_time="201500",
        description="CT HEAD WO CONTRAST",
        indication="Known subdural, follow-up; decreased responsiveness",
        series_desc="AXIAL 5MM",
    ),
}
PATIENT = dict(name="DEMO^SUBDURAL", id="DEMO-0001",
               birth="19510312", sex="M")


def _hu_to_stored(sl):
    return np.clip(sl + 1024, 0, 4095).astype(np.uint16)   # intercept -1024


def write_study(vol, out_dir, which):
    m = STUDY_META[which]
    dest = os.path.join(out_dir, which)
    os.makedirs(dest, exist_ok=True)
    n_sl = vol.shape[0]
    for i in range(n_sl):
        ds = _slice_dataset(vol[i], i, n_sl, m)
        ds.save_as(os.path.join(dest, f"CT{i:03d}.dcm"), enforce_file_format=True)
    return dict(which=which, path=dest, n_slices=n_sl,
                patient_id=PATIENT["id"], **m)


def _slice_dataset(sl, i, n_sl, m):
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = CTImageStorage
    inst_uid = f"{m['series_uid']}.{i+1}"
    fm.MediaStorageSOPInstanceUID = inst_uid
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = generate_uid()

    ds = FileDataset(None, {}, file_meta=fm, preamble=b"\0" * 128)
    ds.SpecificCharacterSet = "ISO_IR 100"
    # patient / study / series
    ds.PatientName = PATIENT["name"]
    ds.PatientID = PATIENT["id"]
    ds.PatientBirthDate = PATIENT["birth"]
    ds.PatientSex = PATIENT["sex"]
    ds.StudyInstanceUID = m["study_uid"]
    ds.SeriesInstanceUID = m["series_uid"]
    ds.StudyDate = ds.SeriesDate = ds.AcquisitionDate = m["study_date"]
    ds.StudyTime = ds.SeriesTime = m["study_time"]
    ds.AccessionNumber = m["accession"]
    ds.StudyID = "1"
    ds.StudyDescription = m["description"]
    ds.SeriesDescription = m["series_desc"]
    ds.SeriesNumber = "1"
    ds.Modality = "CT"
    ds.BodyPartExamined = "HEAD"
    ds.Manufacturer = "SYNTHETIC-DEMO"
    ds.ManufacturerModelName = "phantom-generator"
    # non-contrast markers (help NCCT series selection)
    ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.ContrastBolusAgent = ""
    ds.KVP = "120"
    # image geometry
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = inst_uid
    ds.InstanceNumber = str(i + 1)
    z = i * SLICE_MM
    ds.ImagePositionPatient = [-FOV_MM / 2, -FOV_MM / 2, float(z)]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.SliceLocation = float(z)
    ds.SliceThickness = SLICE_MM
    ds.PixelSpacing = [round(PX, 4), round(PX, 4)]
    ds.FrameOfReferenceUID = f"{m['study_uid']}.9"
    # pixels
    stored = _hu_to_stored(sl)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns = stored.shape
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.RescaleIntercept = "-1024"
    ds.RescaleSlope = "1"
    ds.RescaleType = "HU"
    ds.WindowCenter = "40"
    ds.WindowWidth = "80"
    ds.PixelData = stored.tobytes()
    return ds


if __name__ == "__main__":
    import synthetic_ct
    here = os.path.dirname(__file__)
    meta = synthetic_ct.build_demo_case(os.path.join(here, "pacs"))
    # verify round-trip HU on the follow-up center slice
    fu = os.path.join(here, "pacs", "followup", "CT001.dcm")
    d = pydicom.dcmread(fu)
    hu = d.pixel_array.astype(float) * float(d.RescaleSlope) + \
        float(d.RescaleIntercept)
    print(f"[dicom_writer] round-trip check {fu}: "
          f"blood≈{hu[(hu>55)&(hu<70)].size} px, HU range "
          f"[{hu.min():.0f},{hu.max():.0f}]")
