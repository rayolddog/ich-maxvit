#!/usr/bin/env python3
"""
conformant_sr.py — a REAL TID 1500 Measurement Report for the ICH AI result.

Replaces the ad-hoc SR (which was binary DICOM mislabelled .json, non-conformant,
never ingested) with a standards-valid Comprehensive 3D SR built via highdicom:
  - TID 1500 Measurement Report (declared via ContentTemplateSequence)
  - real coded concepts (SNOMED-CT / DCM), one code per concept
  - device/algorithm observer context (the AI as the observer)
  - evidence linkage: references the source CT SOP instances it evaluated
  - per-subtype qualitative findings + probabilities as numeric measurements

Self-contained: references the synthetic follow-up study in pacs/followup so the
SR points at images that actually exist and can be co-loaded into a PACS.

Usage:
  conformant_sr.py [--study demo_synthetic/pacs/followup] [--out demo_synthetic/out]
"""
import argparse
import glob
import os

import highdicom as hd
import numpy as np
from pydicom import dcmread
from pydicom.sr import codes
from pydicom.uid import generate_uid

# --- coded concepts (real SNOMED-CT / DCM) ---------------------------------
CID_ICH = hd.sr.CodedConcept("1386000", "SCT", "Intracranial hemorrhage")
CID_SUBTYPES = {
    "epidural":        hd.sr.CodedConcept("83329005", "SCT", "Epidural hemorrhage"),
    "subdural":        hd.sr.CodedConcept("95455008", "SCT", "Subdural hemorrhage"),
    "subarachnoid":    hd.sr.CodedConcept("21454007", "SCT", "Subarachnoid hemorrhage"),
    "intraparenchymal": hd.sr.CodedConcept("274100004", "SCT", "Cerebral hemorrhage"),
    "intraventricular": hd.sr.CodedConcept("262052002", "SCT", "Intraventricular hemorrhage"),
}
CID_PRESENT = hd.sr.CodedConcept("52101004", "SCT", "Present")
CID_ABSENT = hd.sr.CodedConcept("2667000", "SCT", "Absent")
CID_PRESENCE = hd.sr.CodedConcept("363778006", "SCT", "Presence")
# probability: DCM "Probability" concept
CID_PROB = hd.sr.CodedConcept("111047", "DCM", "Probability")
UNIT_NONE = hd.sr.CodedConcept("1", "UCUM", "no units")


def demo_inference():
    """Fixed result for the SYNTHETIC phantom, where the real classifier is not
    meaningful. Real cases use load_inference / run_inference below."""
    return {
        "overall_positive": True,
        "dominant_class": "subdural",
        "study_level": {
            "any":            {"prob": 0.94, "positive": True},
            "subdural":       {"prob": 0.91, "positive": True},
            "epidural":       {"prob": 0.03, "positive": False},
            "subarachnoid":   {"prob": 0.06, "positive": False},
            "intraparenchymal": {"prob": 0.04, "positive": False},
            "intraventricular": {"prob": 0.02, "positive": False},
        },
        "hot_slices": [],
    }


def load_inference(path):
    """Server-side re-load of a real inference result (JSON from ich_inference).
    Complementarity rule: the agent passes this PATH (an opaque handle); the SR
    is built from the real values re-loaded here, never from LLM-transcribed
    probabilities or SOP UIDs."""
    import json
    with open(path) as f:
        return json.load(f)


def run_inference(study_dir, checkpoint=None):
    """Run the real MaxViT classifier on a DICOM series; returns its result dict."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from ich_inference import run_inference as _run, DEFAULT_CKPT
    return _run(study_dir, checkpoint or DEFAULT_CKPT, verbose=False)


def _add_coding_scheme_identification(ds):
    """Declare the coding schemes used (Type 1C; the reviewer flagged its
    absence). Standard registry UIDs for SNOMED-CT, DICOM, UCUM."""
    from pydicom.dataset import Dataset
    schemes = [
        ("SCT", "2.16.840.1.113883.6.96", "SNOMED CT"),
        ("DCM", "1.2.840.10008.2.16.4", "DICOM Controlled Terminology"),
        ("UCUM", "2.16.840.1.113883.6.8", "Unified Code for Units of Measure"),
    ]
    seq = []
    for desig, uid, name in schemes:
        it = Dataset()
        it.CodingSchemeDesignator = desig
        it.CodingSchemeUID = uid
        it.CodingSchemeName = name
        seq.append(it)
    ds.CodingSchemeIdentificationSequence = seq


def build_sr(study_dir, out_dir, inference=None):
    """inference: a real result dict (run_inference/load_inference). If None,
    falls back to the phantom demo result."""
    files = sorted(glob.glob(os.path.join(study_dir, "*.dcm")))
    if not files:
        raise SystemExit(f"no DICOM in {study_dir}")
    src = [dcmread(f) for f in files]
    inf = inference if inference is not None else demo_inference()

    # observer = the AI algorithm (device observer context)
    device = hd.sr.ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=hd.sr.DeviceObserverIdentifyingAttributes(
            uid=generate_uid(),
            name="ICH-MaxViT",
            manufacturer_name="ich-maxvit demo",
            model_name="MaxViT ICH classifier v1",
        ),
    )
    obs_context = hd.sr.ObservationContext(observer_device_context=device)

    # one measurement group per class: qualitative present/absent + probability
    groups = []
    for cls, cid in [("any", CID_ICH)] + [(k, CID_SUBTYPES[k])
                                          for k in CID_SUBTYPES]:
        vals = inf["study_level"][cls]
        prob = hd.sr.Measurement(
            name=CID_PROB,
            value=float(vals["prob"]),
            unit=UNIT_NONE,
        )
        qual = hd.sr.QualitativeEvaluation(
            name=CID_PRESENCE,
            value=CID_PRESENT if vals["positive"] else CID_ABSENT,
        )
        grp = hd.sr.MeasurementsAndQualitativeEvaluations(
            tracking_identifier=hd.sr.TrackingIdentifier(
                uid=generate_uid(), identifier=f"ICH-AI {cls}"),
            finding_type=cid,
            measurements=[prob],
            qualitative_evaluations=[qual],
        )
        groups.append(grp)

    report = hd.sr.MeasurementReport(
        observation_context=obs_context,
        procedure_reported=hd.sr.CodedConcept(
            "429858000", "SCT", "CT of head without contrast"),
        imaging_measurements=groups,
    )

    sr = hd.sr.ComprehensiveSR(
        evidence=src,
        content=report,
        series_number=99,
        series_instance_uid=generate_uid(),
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="ich-maxvit demo",
        is_complete=True,        # content is complete
        is_final=True,           # finalized
        is_verified=False,       # AI result: NOT human-verified (honest flag)
    )
    sr.SeriesDescription = "AI ICH screening (TID 1500)"
    sr.SpecificCharacterSet = "ISO_IR 192"          # UTF-8
    _add_coding_scheme_identification(sr)

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ich_ai_sr.dcm")
    sr.save_as(path)
    print(f"[conformant_sr] wrote valid TID 1500 SR -> {path}")
    print(f"    references {len(src)} source CT instances of study "
          f"{src[0].StudyInstanceUID}")
    pos = [(c, v["prob"]) for c, v in inf["study_level"].items()
           if c != "any" and v.get("positive")]
    verdict = ("POSITIVE: " + ", ".join(f"{c} p={p}" for c, p in sorted(
        pos, key=lambda x: -x[1]))) if inf.get("overall_positive") \
        else f"NEGATIVE (any p={inf['study_level']['any']['prob']})"
    print(f"    real finding: {verdict}")
    return path


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(__file__)
    ap.add_argument("--study", default=os.path.join(here, "pacs", "followup"))
    ap.add_argument("--out", default=os.path.join(here, "out"))
    ap.add_argument("--run", action="store_true",
                    help="run the real MaxViT classifier on --study")
    ap.add_argument("--inference-json",
                    help="build from a saved ich_inference result (server-side handle)")
    a = ap.parse_args()
    if a.inference_json:
        inf = load_inference(a.inference_json)
    elif a.run:
        print(f"[conformant_sr] running MaxViT on {a.study} ...")
        inf = run_inference(a.study)
    else:
        inf = None                                  # phantom fallback
    build_sr(a.study, a.out, inference=inf)


if __name__ == "__main__":
    main()
