#!/usr/bin/env python3
"""
ICH DICOM Structured Report Generator.

Produces a Comprehensive SR (SOP 1.2.840.10008.5.1.4.1.1.88.33) encoding:
  - AI method description
  - Per-class ICH probabilities
  - IMAGE references to hot slices (by SOPInstanceUID)
  - PPV / NPV at assumed prevalence
  - Confusion matrix (TP / FP / TN / FN at 1,000 cases)
  - Pain index (FP : TP ratio)
  - Disclaimer

The SR is a separate DICOM object that travels with the study without
altering the diagnostic images.  PACS viewers display it alongside
the source images; radiologists can open it for AI context without
it obscuring the diagnostic series.

Usage (standalone):
    python ich_dicom_sr.py results.json --output ich_ai_sr.dcm \
        --prevalence 0.02 --study-uid 1.2.3 --patient-id P001
"""

import os
import sys
import json
import argparse
import datetime
from pathlib import Path

import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

# SOP classes
SR_COMPREHENSIVE_SOP   = "1.2.840.10008.5.1.4.1.1.88.33"
CT_IMAGE_SOP           = "1.2.840.10008.5.1.4.1.1.2"
VERIFICATION_SOP       = "1.2.840.10008.5.1.4.1.1.88.11"

# Default model constants — overridden at import time by load_test_metrics()
MODEL_SENSITIVITY = 0.98
MODEL_SPECIFICITY = 0.97
MODEL_MEAN_AUC    = 0.9877
MODEL_NAME        = "MaxViT-Base (timm), 118.7M parameters"
TRAINING_DATA     = "RSNA ICH dataset — 544,685 images, 6-class multi-label"
# Per-class metrics keyed by LABEL_COLS name (populated by load_test_metrics)
MODEL_PER_CLASS:  dict = {}
MODEL_METRICS_SOURCE: str = "defaults (run evaluate_maxvit_test.py to update)"

_DEFAULT_METRICS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints_maxvit", "test_metrics.json",
)


def load_test_metrics(path: str = _DEFAULT_METRICS_PATH) -> bool:
    """
    Load per-class test-set metrics from evaluate_maxvit_test.py output.
    Updates module-level MODEL_* constants.  Returns True on success.
    """
    global MODEL_SENSITIVITY, MODEL_SPECIFICITY, MODEL_MEAN_AUC
    global MODEL_PER_CLASS, MODEL_METRICS_SOURCE

    if not os.path.isfile(path):
        return False
    try:
        with open(path) as f:
            data = json.load(f)
        per_class = data.get("per_class", {})
        if not per_class:
            return False

        MODEL_PER_CLASS = per_class
        MODEL_MEAN_AUC  = round(data.get("mean_auc", MODEL_MEAN_AUC), 4)

        # Use "any" class metrics as the headline sensitivity/specificity
        any_m = per_class.get("any", {})
        if any_m:
            MODEL_SENSITIVITY = round(any_m.get("sensitivity", MODEL_SENSITIVITY), 4)
            MODEL_SPECIFICITY = round(any_m.get("specificity", MODEL_SPECIFICITY), 4)

        MODEL_METRICS_SOURCE = (
            f"held-out test set ({data.get('n_test', '?'):,} slices), "
            f"epoch {data.get('epoch', '?')}"
        )
        return True
    except Exception:
        return False


# Attempt to load at import — silently uses defaults if file not yet generated
load_test_metrics()

CLASS_DISPLAY = {
    "epidural":         "Epidural hematoma",
    "intraparenchymal": "Intraparenchymal hemorrhage",
    "intraventricular": "Intraventricular hemorrhage",
    "subarachnoid":     "Subarachnoid hemorrhage",
    "subdural":         "Subdural hematoma",
    "any":              "Any intracranial hemorrhage",
}


# ── Low-level SR helpers ──────────────────────────────────────────────────────

def _code_item(value: str, scheme: str, meaning: str) -> Dataset:
    """Build a CodeSequence item (used for ConceptNameCodeSequence etc.)."""
    ds = Dataset()
    ds.CodeValue               = value
    ds.CodingSchemeDesignator  = scheme
    ds.CodeMeaning             = meaning
    return ds


def _text_item(
    relationship: str,
    concept_value: str, concept_scheme: str, concept_meaning: str,
    text: str,
) -> Dataset:
    """Build a TEXT content item."""
    ds = Dataset()
    ds.RelationshipType         = relationship
    ds.ValueType                = "TEXT"
    ds.ConceptNameCodeSequence  = Sequence([
        _code_item(concept_value, concept_scheme, concept_meaning)
    ])
    ds.TextValue                = text
    return ds


def _num_item(
    relationship: str,
    concept_value: str, concept_scheme: str, concept_meaning: str,
    numeric_value: float,
    unit_value: str, unit_scheme: str, unit_meaning: str,
) -> Dataset:
    """Build a NUM content item."""
    ds = Dataset()
    ds.RelationshipType         = relationship
    ds.ValueType                = "NUM"
    ds.ConceptNameCodeSequence  = Sequence([
        _code_item(concept_value, concept_scheme, concept_meaning)
    ])
    measured = Dataset()
    measured.NumericValue       = str(round(numeric_value, 4))
    measured.MeasurementUnitsCodeSequence = Sequence([
        _code_item(unit_value, unit_scheme, unit_meaning)
    ])
    ds.MeasuredValueSequence    = Sequence([measured])
    return ds


def _image_item(
    relationship: str,
    sop_instance_uid: str,
    sop_class_uid:    str = CT_IMAGE_SOP,
) -> Dataset:
    """Build an IMAGE content item referencing a specific DICOM instance."""
    ref = Dataset()
    ref.ReferencedSOPClassUID    = sop_class_uid
    ref.ReferencedSOPInstanceUID = sop_instance_uid

    ds = Dataset()
    ds.RelationshipType         = relationship
    ds.ValueType                = "IMAGE"
    ds.ConceptNameCodeSequence  = Sequence([
        _code_item("121233", "DCM", "Source Image")
    ])
    ds.ReferencedSOPSequence    = Sequence([ref])
    return ds


def _container(
    relationship: str,
    concept_value: str, concept_scheme: str, concept_meaning: str,
    continuity: str = "SEPARATE",
) -> Dataset:
    """Build a CONTAINER content item (holds child items via ContentSequence)."""
    ds = Dataset()
    ds.RelationshipType         = relationship
    ds.ValueType                = "CONTAINER"
    ds.ConceptNameCodeSequence  = Sequence([
        _code_item(concept_value, concept_scheme, concept_meaning)
    ])
    ds.ContinuityOfContent      = continuity
    ds.ContentSequence          = Sequence([])
    return ds


# ── Performance metric computation ───────────────────────────────────────────

def compute_metrics(prevalence: float, n_cases: int = 1000) -> dict:
    """
    Compute PPV, NPV, confusion matrix, and pain index at a given prevalence.
    Uses sensitivity/specificity from test-set evaluation if available,
    otherwise falls back to training-era defaults.
    """
    n_pos      = round(n_cases * prevalence)
    n_neg      = n_cases - n_pos
    tp         = round(n_pos * MODEL_SENSITIVITY)
    fn         = n_pos - tp
    fp         = round(n_neg * (1 - MODEL_SPECIFICITY))
    tn         = n_neg - fp
    ppv        = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv        = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    pain_index = round(fp / tp, 2) if tp > 0 else float("inf")

    # LR+ and LR- from test metrics if available
    any_m  = MODEL_PER_CLASS.get("any", {})
    lr_pos = round(any_m.get("lr_positive", MODEL_SENSITIVITY / max(1 - MODEL_SPECIFICITY, 1e-10)), 2)
    lr_neg = round(any_m.get("lr_negative", (1 - MODEL_SENSITIVITY) / max(MODEL_SPECIFICITY, 1e-10)), 4)

    return {
        "prevalence":    prevalence,
        "n_cases":       n_cases,
        "sensitivity":   MODEL_SENSITIVITY,
        "specificity":   MODEL_SPECIFICITY,
        "mean_auc":      MODEL_MEAN_AUC,
        "lr_positive":   lr_pos,
        "lr_negative":   lr_neg,
        "ppv":           round(ppv,  4),
        "npv":           round(npv,  4),
        "pain_index":    pain_index,
        "tp": tp, "fn": fn,
        "fp": fp, "tn": tn,
        "metrics_source": MODEL_METRICS_SOURCE,
        "per_class":     MODEL_PER_CLASS,
    }


# ── SR content tree construction ──────────────────────────────────────────────

def _build_content_tree(
    inference:      dict,
    study_metadata: dict,
    metrics:        dict,
) -> list:
    """
    Build the SR content sequence (list of top-level content items).
    Structure:
      Root document title (implicit)
        CONTAINER: AI Method
        CONTAINER: ICH Screening Result
        CONTAINER: Referenced Images (hot slices)
        CONTAINER: Performance Metrics at Assumed Prevalence
        TEXT:      Disclaimer
    """
    content = []

    # ── 1. AI Method ──────────────────────────────────────────────────────────
    ai_method = _container(
        "CONTAINS",
        "111001", "DCM", "Algorithm Name",
        continuity="SEPARATE",
    )
    ai_method.ContentSequence = Sequence([
        _text_item("CONTAINS",
                   "111001", "DCM", "Algorithm Name",
                   MODEL_NAME),
        _text_item("CONTAINS",
                   "111003", "DCM", "Algorithm Version",
                   f"Mean AUC {MODEL_MEAN_AUC:.4f} — {MODEL_METRICS_SOURCE}"),
        _text_item("CONTAINS",
                   "111526", "DCM", "Image Library",
                   TRAINING_DATA),
        _text_item("CONTAINS",
                   "121106", "DCM", "Key Images",
                   f"Evaluated {inference.get('valid_slices', 0)} axial NCCT slices"),
    ])
    content.append(ai_method)

    # ── 2. ICH Screening Result ───────────────────────────────────────────────
    overall_positive = inference.get("overall_positive", False)
    dominant         = inference.get("dominant_class", "")

    result_container = _container(
        "CONTAINS",
        "121071", "DCM", "Finding",
        continuity="SEPARATE",
    )

    result_items = [
        _text_item("CONTAINS",
                   "121071", "DCM", "Finding",
                   ("POSITIVE — " + dominant) if overall_positive else "NEGATIVE"),
    ]

    # Per-class probabilities
    study_level = inference.get("study_level", {})
    for col, vals in study_level.items():
        display = CLASS_DISPLAY.get(col, col)
        prob    = vals.get("prob", 0.0)
        call    = "POSITIVE" if vals.get("positive") else "negative"
        result_items.append(
            _text_item("CONTAINS",
                       "121071", "DCM", display,
                       f"{display}: score {prob*100:.1f}% ({call})")
        )

    result_container.ContentSequence = Sequence(result_items)
    content.append(result_container)

    # ── 3. Referenced Images (hot slices) ─────────────────────────────────────
    hot_slices = inference.get("hot_slices", [])
    if hot_slices:
        img_container = _container(
            "CONTAINS",
            "121130", "DCM", "Patient Exposure",   # reused as image ref section
            continuity="SEPARATE",
        )
        img_items = [
            _text_item("CONTAINS",
                       "121106", "DCM", "Key Images",
                       "Slices with highest AI activation for detected ICH class. "
                       "Spatial localization is approximate (slice-level model)."),
        ]
        for s in hot_slices:
            sop_uid = s.get("sop_uid", "")
            if sop_uid:
                img_items.append(_image_item("CONTAINS", sop_uid))
            img_items.append(
                _text_item("CONTAINS",
                           "112039", "DCM", "Tracking Identifier",
                           f"Slice {s['slice_index']} | "
                           f"z={s.get('slice_z_mm', '?'):+} mm | "
                           f"{CLASS_DISPLAY.get(s['dominant_class'], s['dominant_class'])} | "
                           f"score {s['prob']*100:.1f}%")
            )
        img_container.ContentSequence = Sequence(img_items)
        content.append(img_container)

    # ── 4. Performance Metrics at Assumed Prevalence ──────────────────────────
    m = metrics
    perf_container = _container(
        "CONTAINS",
        "C67447", "NCIt", "Performance",
        continuity="SEPARATE",
    )

    confusion_text = (
        f"At assumed prevalence {m['prevalence']*100:.0f}%, "
        f"per {m['n_cases']:,} cases:\n"
        f"\n"
        f"              AI Positive   AI Negative\n"
        f"  True ICH    {m['tp']:>11}   {m['fn']:>11}   (TP={m['tp']}, FN={m['fn']})\n"
        f"  No ICH      {m['fp']:>11}   {m['tn']:>11}   (FP={m['fp']}, TN={m['tn']})\n"
    )

    pain_text = (
        f"Pain Index (FP:TP ratio) = {m['pain_index']:.2f} "
        f"at {m['prevalence']*100:.0f}% prevalence: "
        f"for every {m['tp']} true ICH detections, the radiologist must "
        f"review and override {m['fp']} false positive AI flags. "
        f"Each override represents additional review time and cognitive load "
        f"required to reject an incorrect AI positive call."
    )

    lr_pos_str = f"{m['lr_positive']:.2f}" if m.get("lr_positive", 0) < 1e5 else "∞"
    lr_neg_str = f"{m.get('lr_negative', 0):.4f}"

    perf_items = [
        _text_item("CONTAINS",
                   "C25429", "NCIt", "Assumed Prevalence",
                   f"{m['prevalence']*100:.0f}% "
                   f"({study_metadata.get('indication', 'not specified')})"),
        _text_item("CONTAINS",
                   "C0242960", "UMLS", "Metrics Source",
                   m.get("metrics_source", "defaults")),
        _num_item("CONTAINS",
                  "C41189", "NCIt", "Sensitivity",
                  m["sensitivity"],
                  "%", "UCUM", "Percent"),
        _num_item("CONTAINS",
                  "C41190", "NCIt", "Specificity",
                  m["specificity"],
                  "%", "UCUM", "Percent"),
        _num_item("CONTAINS",
                  "C25651", "NCIt", "Mean AUC",
                  m.get("mean_auc", MODEL_MEAN_AUC),
                  "{1}", "UCUM", "ratio"),
        _num_item("CONTAINS",
                  "C41192", "NCIt", "Positive Predictive Value",
                  m["ppv"],
                  "%", "UCUM", "Percent"),
        _num_item("CONTAINS",
                  "C41194", "NCIt", "Negative Predictive Value",
                  m["npv"],
                  "%", "UCUM", "Percent"),
        _text_item("CONTAINS",
                   "C0392762", "UMLS", "Positive Likelihood Ratio",
                   f"LR+ = {lr_pos_str}"),
        _text_item("CONTAINS",
                   "C0392762", "UMLS", "Negative Likelihood Ratio",
                   f"LR- = {lr_neg_str}"),
        _text_item("CONTAINS",
                   "C0009401", "UMLS", "Confusion Matrix",
                   confusion_text),
        _text_item("CONTAINS",
                   "C0030705", "UMLS", "Pain Index",
                   pain_text),
    ]
    perf_container.ContentSequence = Sequence(perf_items)
    content.append(perf_container)

    # ── 5. Disclaimer ─────────────────────────────────────────────────────────
    content.append(
        _text_item("CONTAINS",
                   "121106", "DCM", "Key Images",
                   "AI-assisted screening tool. Not FDA-cleared for clinical use. "
                   "All findings require radiologist confirmation. "
                   "DICOM Structured Report is the appropriate vehicle for "
                   "interactive AI finding review in a production implementation.")
    )

    return content


# ── Top-level SR builder ──────────────────────────────────────────────────────

def generate_sr(
    inference_results: dict,
    study_metadata:    dict,
    output_path:       str,
    prevalence:        float = 0.02,
) -> dict:
    """
    Generate a DICOM Comprehensive SR file.

    Args:
        inference_results : dict from ich_inference.run_inference()
        study_metadata    : dict with keys: study_uid, patient_id,
                            patient_name (optional), indication (optional),
                            series_uid (optional, source series)
        output_path       : path to write the .dcm SR file
        prevalence        : assumed disease prevalence for metrics (0.0–1.0)

    Returns:
        dict with status, output_path, and computed metrics
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.datetime.now()
    date_str = now.strftime("%Y%m%d")
    time_str = now.strftime("%H%M%S.%f")

    metrics = compute_metrics(prevalence)

    # ── File meta ─────────────────────────────────────────────────────────────
    file_meta = pydicom.Dataset()
    file_meta.MediaStorageSOPClassUID    = SR_COMPREHENSIVE_SOP
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

    # ── Main dataset ──────────────────────────────────────────────────────────
    ds = FileDataset(
        str(output_path),
        {},
        file_meta  = file_meta,
        preamble   = b"\x00" * 128,
    )
    ds.is_implicit_VR = False
    ds.is_little_endian = True

    # Patient
    ds.PatientID   = study_metadata.get("patient_id",   "UNKNOWN")
    ds.PatientName = study_metadata.get("patient_name", "UNKNOWN^UNKNOWN")
    ds.PatientBirthDate = ""
    ds.PatientSex       = ""

    # Study
    ds.StudyInstanceUID  = study_metadata.get("study_uid", generate_uid())
    ds.StudyDate         = date_str
    ds.StudyTime         = time_str
    ds.StudyDescription  = "AI ICH Screening Report"
    ds.AccessionNumber   = ""
    ds.ReferringPhysicianName = ""

    # SR Series
    ds.SeriesInstanceUID   = generate_uid()
    ds.SeriesNumber        = "99"
    ds.SeriesDescription   = "AI ICH SR"
    ds.Modality            = "SR"
    ds.SeriesDate          = date_str
    ds.SeriesTime          = time_str

    # SOP
    ds.SOPClassUID         = SR_COMPREHENSIVE_SOP
    ds.SOPInstanceUID      = file_meta.MediaStorageSOPInstanceUID
    ds.InstanceNumber      = "1"
    ds.ContentDate         = date_str
    ds.ContentTime         = time_str

    # SR document attributes
    ds.ValueType                  = "CONTAINER"
    ds.ContinuityOfContent        = "SEPARATE"
    ds.CompletionFlag             = "COMPLETE"
    ds.VerificationFlag           = "UNVERIFIED"
    ds.ConceptNameCodeSequence    = Sequence([
        _code_item("126000", "DCM", "Imaging Measurement Report")
    ])

    # ── Content tree ──────────────────────────────────────────────────────────
    ds.ContentSequence = Sequence(
        _build_content_tree(inference_results, study_metadata, metrics)
    )

    # ── Write file ────────────────────────────────────────────────────────────
    pydicom.dcmwrite(str(output_path), ds)

    result = {
        "status":       "DICOM SR written",
        "output_path":  str(output_path),
        "sop_instance": ds.SOPInstanceUID,
        "metrics":      metrics,
    }
    print(f"\n[SR] Written → {output_path}")
    print(f"     Metrics : {metrics.get('metrics_source', 'defaults')}")
    print(f"     AUC {metrics.get('mean_auc', MODEL_MEAN_AUC):.4f}  "
          f"Sens {metrics['sensitivity']:.3f}  Spec {metrics['specificity']:.3f}  "
          f"LR+ {metrics.get('lr_positive', '?')}  LR- {metrics.get('lr_negative', '?')}")
    print(f"     PPV {metrics['ppv']*100:.1f}%  NPV {metrics['npv']*100:.1f}%  "
          f"Pain index {metrics['pain_index']:.2f}  "
          f"(prevalence {prevalence*100:.0f}%)")
    return result


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate DICOM Comprehensive SR from ICH inference results"
    )
    parser.add_argument("results_json",
                        help="JSON file from ich_inference.py (--output-json)")
    parser.add_argument("--output",     default="ich_ai_sr.dcm",
                        help="Output .dcm SR file path")
    parser.add_argument("--prevalence", type=float, default=0.02,
                        help="Assumed prevalence (0.0–1.0); default 0.02")
    parser.add_argument("--study-uid",  default="",
                        help="DICOM StudyInstanceUID")
    parser.add_argument("--patient-id", default="UNKNOWN",
                        help="Patient ID")
    parser.add_argument("--indication", default="",
                        help="Clinical indication text")
    args = parser.parse_args()

    with open(args.results_json) as f:
        inference = json.load(f)

    study_metadata = {
        "study_uid":   args.study_uid,
        "patient_id":  args.patient_id,
        "indication":  args.indication,
    }

    generate_sr(
        inference_results = inference,
        study_metadata    = study_metadata,
        output_path       = args.output,
        prevalence        = args.prevalence,
    )


if __name__ == "__main__":
    main()
