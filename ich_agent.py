#!/usr/bin/env python3
"""
ICH Detection Agent — Anthropic tool-use pipeline.

Accepts a DICOM study folder, identifies the axial noncontrast CT head
sequence, runs MaxViT ICH inference, generates a DICOM SR, flags the
worklist if positive, and composes a report paragraph + impression bullet.

Usage:
    python ich_agent.py /path/to/dicom/study --indication "headache"
    python ich_agent.py /path/to/dicom/study --indication "trauma" --prevalence 0.10
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Any

import anthropic

try:
    from ich_worklist import record_result as _worklist_record
    _WORKLIST_AVAILABLE = True
except ImportError:
    _WORKLIST_AVAILABLE = False

try:
    from prevalence_db import PrevalenceDB, normalize_location
    _PREVALENCE_DB = PrevalenceDB()
    _PREVALENCE_DB_AVAILABLE = True
except ImportError:
    _PREVALENCE_DB_AVAILABLE = False


# ── Tool implementations ────────────────────────────────────────────────────

def _scan_study(study_folder: str) -> dict:
    """
    Scan a DICOM study folder and return metadata for every series found.
    Reads only DICOM headers (no pixel data) for speed.
    """
    import pydicom

    study_folder = Path(study_folder)
    if not study_folder.exists():
        return {"error": f"Study folder not found: {study_folder}"}

    series: dict[str, dict] = {}

    for dcm_file in sorted(study_folder.rglob("*.dcm")):
        try:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
            if uid not in series:
                series[uid] = {
                    "series_uid":           uid,
                    "series_number":        str(getattr(ds, "SeriesNumber",        "")),
                    "series_description":   str(getattr(ds, "SeriesDescription",   "")),
                    "modality":             str(getattr(ds, "Modality",            "")),
                    "image_type":           list(getattr(ds, "ImageType",          [])),
                    "patient_position":     str(getattr(ds, "PatientPosition",     "")),
                    "slice_thickness":      str(getattr(ds, "SliceThickness",      "")),
                    "contrast_bolus_agent": str(getattr(ds, "ContrastBolusAgent",  "")),
                    "kvp":                  str(getattr(ds, "KVP",                 "")),
                    "study_description":    str(getattr(ds, "StudyDescription",    "")),
                    "study_uid":            str(getattr(ds, "StudyInstanceUID",    "")),
                    "patient_id":           str(getattr(ds, "PatientID",           "")),
                    "institutional_department": str(getattr(ds, "InstitutionalDepartmentName", "")),
                    "series_folder":        str(dcm_file.parent),
                    "slice_count":          0,
                }
            series[uid]["slice_count"] += 1

        except Exception:
            continue

    return {
        "study_folder":  str(study_folder),
        "series_count":  len(series),
        "series":        list(series.values()),
    }


def _run_ich_inference(series_folder: str, checkpoint_path: str) -> dict:
    """
    Run MaxViT ICH inference on a DICOM series folder.
    Delegates to ich_inference.run_inference().
    """
    from ich_inference import run_inference
    result = run_inference(
        series_folder   = series_folder,
        checkpoint_path = checkpoint_path,
        verbose         = True,
    )
    # Drop per_slice_probs from agent context — too large for message history
    result.pop("per_slice_probs", None)
    return result


def _generate_dicom_sr(
    inference_results: dict,
    study_metadata:    dict,
    output_path:       str,
    prevalence:        float = 0.02,
) -> dict:
    """Generate a DICOM Comprehensive SR. Delegates to ich_dicom_sr.generate_sr()."""
    from ich_dicom_sr import generate_sr
    return generate_sr(
        inference_results = inference_results,
        study_metadata    = study_metadata,
        output_path       = output_path,
        prevalence        = prevalence,
    )


def _flag_worklist(
    study_uid:      str,
    positive:       bool,
    dominant_class: str = "",
) -> dict:
    """
    Simulate flagging a study on the radiology worklist.
    In production: sends HL7 ORM message or calls PACS API.
    In demonstration: prints console notice and returns flag record.
    """
    flag = {
        "study_uid":       study_uid,
        "ai_priority_flag": positive,
        "flag_type":       "AI_ICH_DETECTED" if positive else "AI_ICH_NEGATIVE",
        "flag_reason":     (
            f"AI-detected {dominant_class}" if positive
            else "AI screening negative — routine read"
        ),
        "simulated":       True,
    }

    marker = "⚠  PRIORITY FLAG SET" if positive else "✓  No priority flag"
    print(f"\n[WORKLIST] {marker}: {flag['flag_reason']}")
    return flag


# ── Tool registry ────────────────────────────────────────────────────────────

CHECKPOINT_PATH = (
    "/home/justinolddog/NewICH/checkpoints_maxvit/best_maxvit_ich.pth"
)

TOOL_IMPLEMENTATIONS = {
    "scan_study": lambda a: _scan_study(
        a["study_folder"]
    ),
    "run_ich_inference": lambda a: _run_ich_inference(
        a["series_folder"],
        a.get("checkpoint_path", CHECKPOINT_PATH),
    ),
    "generate_dicom_sr": lambda a: _generate_dicom_sr(
        a["inference_results"],
        a["study_metadata"],
        a["output_path"],
        a.get("prevalence", 0.02),
    ),
    "flag_worklist": lambda a: _flag_worklist(
        a["study_uid"],
        a["positive"],
        a.get("dominant_class", ""),
    ),
}


# ── Tool schemas ─────────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "scan_study",
        "description": (
            "Scan a DICOM study folder and return metadata for every series found. "
            "Returns series description, modality, image type, slice thickness, "
            "contrast bolus agent, KVP, and slice count for each series. "
            "Always call this first to understand what sequences are present."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "study_folder": {
                    "type":        "string",
                    "description": "Absolute path to the DICOM study folder.",
                },
            },
            "required": ["study_folder"],
        },
    },
    {
        "name": "run_ich_inference",
        "description": (
            "Run the MaxViT ICH classifier on a confirmed axial noncontrast CT "
            "head series. Returns per-slice probabilities for 6 ICH classes "
            "(epidural, intraparenchymal, intraventricular, subarachnoid, "
            "subdural, any), study-level predictions, hot slices (highest "
            "activation per class), and an overall positive/negative determination. "
            "Only call this after confirming the series is axial NCCT head."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "series_folder": {
                    "type":        "string",
                    "description": "Absolute path to the selected NCCT series folder.",
                },
                "checkpoint_path": {
                    "type":        "string",
                    "description": "Path to the MaxViT .pth checkpoint. Defaults to best_maxvit_ich.pth.",
                },
            },
            "required": ["series_folder"],
        },
    },
    {
        "name": "generate_dicom_sr",
        "description": (
            "Generate a DICOM Structured Report encoding the AI findings. "
            "The SR includes: AI method, per-class probabilities, PPV and NPV "
            "at the assumed prevalence, a confusion matrix (TP/FP/TN/FN at 1000 "
            "cases), the pain index (FP:TP ratio), and references to hot slices. "
            "The SR is written as a separate file alongside the study images."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "inference_results": {
                    "type":        "object",
                    "description": "The full results dict returned by run_ich_inference.",
                },
                "study_metadata": {
                    "type":        "object",
                    "description": "Study-level metadata: study_uid, patient_id, indication.",
                },
                "output_path": {
                    "type":        "string",
                    "description": "Full file path where the SR should be written.",
                },
                "prevalence": {
                    "type":        "number",
                    "description": (
                        "Assumed disease prevalence (0.0–1.0) for PPV/NPV and confusion "
                        "matrix. Guidelines: headache/no trauma ~0.02, altered mental "
                        "status ~0.08, ER trauma ~0.10, post-op craniotomy ~0.20."
                    ),
                },
            },
            "required": ["inference_results", "study_metadata", "output_path"],
        },
    },
    {
        "name": "flag_worklist",
        "description": (
            "Simulate flagging a study on the radiology worklist for priority "
            "AI-detected review. In production this calls a PACS API or sends "
            "an HL7 message. In the demonstration it prints a console notice. "
            "Call this when ICH has been detected (positive=true). The AI-detected "
            "priority flag is visually distinct from a physician-ordered stat flag."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "study_uid": {
                    "type":        "string",
                    "description": "DICOM StudyInstanceUID of the study to flag.",
                },
                "positive": {
                    "type":        "boolean",
                    "description": "True if ICH detected; False to log a negative result.",
                },
                "dominant_class": {
                    "type":        "string",
                    "description": "ICH subtype with highest probability, e.g. 'subdural hematoma'.",
                },
            },
            "required": ["study_uid", "positive"],
        },
    },
]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI radiology pipeline agent. Your task is to \
process a CT head study and screen it for intracranial hemorrhage (ICH) using \
a trained MaxViT classifier (mean AUC 0.988, trained on the RSNA ICH dataset).

Follow this sequence exactly:

1. Call scan_study to inventory all series in the study folder.

2. Review the series metadata and identify the axial noncontrast CT head sequence.
   Selection criteria:
   - Modality: CT
   - No ContrastBolusAgent (empty or absent)
   - ImageType should not contain "DERIVED" as the primary type for source images
   - SeriesDescription containing: head, brain, axial, non-con, without, ncct, w/o
   - Patient position: HFS or HFP typical for head CT
   - If multiple qualifying series: prefer thinnest SliceThickness
   If no qualifying series exists, stop and explain why screening was not performed.

3. Call run_ich_inference on the selected series folder.

4. Select the assumed prevalence based on the clinical indication:
   - Headache, no trauma: 0.02
   - Altered mental status: 0.08
   - ER trauma: 0.10
   - Post-op craniotomy: 0.20
   - Unknown/not provided: 0.05

5. Call generate_dicom_sr with the inference results, study metadata (study_uid,
   patient_id, indication), the output path, and the assumed prevalence.
   Output path: same directory as the study folder, filename: ich_ai_sr.json

6. If ICH was detected (overall_positive=True), call flag_worklist.

7. Write your final output containing exactly two clearly labeled sections:

   REPORT BODY PARAGRAPH:
   [A paragraph for inclusion in the body of a radiology report. State that AI
   screening was performed, which sequence was evaluated (slice count), and the
   result. If positive: name the ICH subtype(s), slice range, anatomical location
   relative to the study, and the PPV at assumed prevalence. If negative: state
   the NPV at assumed prevalence. Always note that radiologist confirmation is
   required.]

   IMPRESSION BULLET POINT:
   [A single bullet point for the Impression section. If positive: name the
   finding and note that the study was flagged for priority AI-detected review.
   If negative: note the high NPV and that routine read is appropriate.]

Use precise radiological terminology. Be concise. Always append:
"AI-assisted screening. Not FDA-cleared. Requires radiologist confirmation."
"""


# ── Agent loop ────────────────────────────────────────────────────────────────

def _parse_report_sections(text: str) -> tuple[str, str]:
    """
    Extract the REPORT BODY PARAGRAPH and IMPRESSION BULLET POINT sections
    from the agent's final text output.  Returns (paragraph, bullet).
    """
    paragraph = ""
    bullet    = ""

    # Find REPORT BODY PARAGRAPH section
    rp_marker = "REPORT BODY PARAGRAPH:"
    ib_marker = "IMPRESSION BULLET POINT:"

    rp_idx = text.upper().find(rp_marker.upper())
    ib_idx = text.upper().find(ib_marker.upper())

    if rp_idx != -1:
        start = rp_idx + len(rp_marker)
        end   = ib_idx if ib_idx > rp_idx else len(text)
        paragraph = text[start:end].strip()

    if ib_idx != -1:
        bullet = text[ib_idx + len(ib_marker):].strip()

    return paragraph, bullet


def run_agent(
    study_folder: str,
    indication:   str  = "",
    verbose:      bool = True,
) -> str:
    """
    Run the ICH detection agent on a DICOM study folder.
    Returns the agent's final text output (report paragraph + impression bullet).
    Results are also persisted to the worklist via ich_worklist.record_result().
    """
    client = anthropic.Anthropic()

    user_message = f"Process the CT study at: {study_folder}"
    if indication:
        user_message += f"\nClinical indication: {indication}"

    messages = [{"role": "user", "content": user_message}]

    if verbose:
        print(f"\n{'='*64}")
        print("ICH Detection Agent")
        print(f"Study     : {study_folder}")
        print(f"Indication: {indication or '(not provided)'}")
        print(f"{'='*64}\n")

    # Accumulate key results from tool calls for worklist recording
    _state: dict = {
        "study_uid":        "",
        "patient_id":       "",
        "description":      "",
        "raw_department":   "",
        "location":         "",
        "prevalence_source": "",
        "inference":        {},
        "sr_path":          "",
        "metrics":          {},
        "prevalence":       0.05,
        "ai_positive":      False,
        "dominant_class":   "",
    }

    # ── Agentic loop ──────────────────────────────────────────────────────────
    while True:
        response = client.messages.create(
            model      = "claude-opus-4-6",
            max_tokens = 4096,
            system     = SYSTEM_PROMPT,
            tools      = TOOLS,
            messages   = messages,
        )

        # Append assistant turn to conversation history
        messages.append({"role": "assistant", "content": response.content})

        if verbose:
            for block in response.content:
                if hasattr(block, "text") and block.text:
                    print(block.text)
                elif block.type == "tool_use":
                    print(f"\n[TOOL CALL] {block.name}")
                    preview = json.dumps(block.input, indent=2)
                    if len(preview) > 400:
                        preview = preview[:400] + "\n  ..."
                    print(f"  Input : {preview}")

        # Done — persist to worklist and return final text
        if response.stop_reason == "end_turn":
            final_text = "\n".join(
                block.text
                for block in response.content
                if hasattr(block, "text") and block.text
            )

            if _state["study_uid"]:
                paragraph, bullet = _parse_report_sections(final_text)

                # Record in prevalence DB (every processed study, win or lose)
                if _PREVALENCE_DB_AVAILABLE:
                    try:
                        _PREVALENCE_DB.record_study(
                            study_uid         = _state["study_uid"],
                            ai_positive       = _state["ai_positive"],
                            patient_id        = _state["patient_id"],
                            raw_department    = _state["raw_department"],
                            dominant_class    = _state["dominant_class"],
                            study_level_probs = _state["inference"].get("study_level", {}),
                            series_folder     = _state["inference"].get("series_folder", ""),
                        )
                        if verbose:
                            loc = _state["location"] or "Unknown"
                            print(f"\n[PREV DB] Study recorded — location={loc}")
                    except Exception as exc:
                        if verbose:
                            print(f"\n[PREV DB] record_study failed: {exc}")

                if _WORKLIST_AVAILABLE:
                    try:
                        _worklist_record(
                            study_uid         = _state["study_uid"],
                            patient_id        = _state["patient_id"],
                            indication        = indication,
                            ai_positive       = _state["ai_positive"],
                            ai_result         = _state["inference"],
                            report_paragraph  = paragraph,
                            impression_bullet = bullet,
                            sr_path           = _state["sr_path"],
                            metrics           = _state["metrics"],
                            prevalence        = _state["prevalence"],
                            description       = _state["description"],
                        )
                        if verbose:
                            flag = "POSITIVE" if _state["ai_positive"] else "negative"
                            print(f"[WORKLIST] Study recorded — AI {flag}")
                    except Exception as exc:
                        if verbose:
                            print(f"[WORKLIST] record_result failed: {exc}")

            return final_text

        if response.stop_reason != "tool_use":
            print(f"[WARNING] Unexpected stop_reason: {response.stop_reason}")
            break

        # Execute each tool call and collect results
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            if verbose:
                print(f"\n[EXECUTING] {block.name}...")

            if block.name in TOOL_IMPLEMENTATIONS:
                try:
                    result        = TOOL_IMPLEMENTATIONS[block.name](block.input)
                    result_json   = json.dumps(result)
                    if verbose:
                        preview = result_json[:400] + "..." if len(result_json) > 400 else result_json
                        print(f"  Result: {preview}")

                    # ── Capture state from tool results ───────────────────────
                    if block.name == "scan_study" and "series" in result:
                        # Grab study_uid, patient_id, description from first series
                        first = next(iter(result["series"]), {})
                        _state["study_uid"]     = first.get("study_uid",  "")
                        _state["patient_id"]    = first.get("patient_id", "")
                        _state["description"]   = first.get("study_description", "")

                        # Extract department and look up local prevalence
                        raw_dept = first.get("institutional_department", "")
                        _state["raw_department"] = raw_dept
                        if _PREVALENCE_DB_AVAILABLE:
                            location = normalize_location(raw_dept)
                            _state["location"] = location
                            local_prev, prev_source = (
                                _PREVALENCE_DB.best_prevalence_for_agent(
                                    location = location or None,
                                    days     = 365,
                                    min_n    = 30,
                                )
                            )
                            if local_prev is not None:
                                _state["prevalence"]        = local_prev
                                _state["prevalence_source"] = prev_source
                                # Inject local prevalence into result so Claude uses it
                                result["local_prevalence"] = local_prev
                                result["local_prevalence_source"] = prev_source
                                result_json = json.dumps(result)
                                if verbose:
                                    print(f"  [DB] Local prevalence ({location}): "
                                          f"{local_prev*100:.2f}%  ({prev_source})")

                    elif block.name == "run_ich_inference" and "overall_positive" in result:
                        _state["inference"]      = result
                        _state["ai_positive"]    = result.get("overall_positive", False)
                        _state["dominant_class"] = result.get("dominant_class", "")

                    elif block.name == "generate_dicom_sr":
                        _state["sr_path"]   = result.get("output_path", "")
                        _state["metrics"]   = result.get("metrics", {})
                        _state["prevalence"] = block.input.get("prevalence", 0.05)

                except Exception as e:
                    result_json = json.dumps({"error": str(e)})
                    if verbose:
                        print(f"  Error : {e}")
            else:
                result_json = json.dumps({"error": f"Unknown tool: {block.name}"})

            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result_json,
            })

        # Feed all tool results back in a single user turn
        messages.append({"role": "user", "content": tool_results})

    return ""


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ICH Detection Agent — Anthropic tool-use pipeline"
    )
    parser.add_argument(
        "study_folder",
        help="Path to DICOM study folder",
    )
    parser.add_argument(
        "--indication", default="",
        help="Clinical indication (e.g. 'headache', 'trauma', 'post-op craniotomy')",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose tool call output; print only final report text",
    )
    args = parser.parse_args()

    result = run_agent(
        study_folder = args.study_folder,
        indication   = args.indication,
        verbose      = not args.quiet,
    )

    print(f"\n{'='*64}")
    print("FINAL AGENT OUTPUT")
    print(f"{'='*64}")
    print(result)


if __name__ == "__main__":
    main()
