#!/usr/bin/env python3
"""
ICH Inference — MaxViT slice-level inference on a DICOM series.

Reads a folder of DICOM files, converts each slice to a medium-window
HU tensor, runs the trained MaxViT classifier, and returns per-slice
probabilities and study-level aggregated results.

Reuses:
  dicom_reader_1ch.py  — DICOM → HU float32
  hu_windows.py        — apply_window / WINDOW_MEDIUM

Usage (standalone):
    python ich_inference.py /path/to/series --checkpoint ./checkpoints_maxvit/best_maxvit_ich.pth
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dicom_reader_1ch import read_dicom_hu
from hu_windows import apply_window, WINDOW_MEDIUM
from dataset_1ch import LABEL_COLS

# ── Constants ────────────────────────────────────────────────────────────────

TARGET_SIZE       = 224
BATCH_SIZE        = 32        # slices per forward pass
POSITIVE_THRESH   = 0.5       # probability threshold for positive call
DEFAULT_CKPT      = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints_maxvit", "best_maxvit_ich.pth"
)

# Human-readable class names matching LABEL_COLS order
CLASS_DISPLAY = {
    "epidural":         "epidural hematoma",
    "intraparenchymal": "intraparenchymal hemorrhage",
    "intraventricular": "intraventricular hemorrhage",
    "subarachnoid":     "subarachnoid hemorrhage",
    "subdural":         "subdural hematoma",
    "any":              "any intracranial hemorrhage",
}


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load the trained MaxViT model from a checkpoint."""
    import timm

    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)

    model = timm.create_model(
        "maxvit_base_tf_224.in1k",
        pretrained  = False,
        in_chans    = 1,
        num_classes = len(LABEL_COLS),
    )

    state = ckpt["model_state_dict"]

    # Handle compiled vs non-compiled key prefix mismatch
    try:
        model.load_state_dict(state)
    except RuntimeError:
        sample_key      = next(iter(state))
        ckpt_has_prefix = sample_key.startswith("_orig_mod.")
        if ckpt_has_prefix:
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)

    model.eval()
    model.to(device)
    print(f"  Model loaded  (epoch {ckpt.get('epoch', '?')}, "
          f"AUC {ckpt.get('best_auc', 0):.4f})")
    return model


# ── DICOM series loading ──────────────────────────────────────────────────────

def load_series_slices(series_folder: str) -> list[dict]:
    """
    Read all valid DICOM slices from a folder.
    Returns a list of dicts sorted by ImagePositionPatient z (superior → inferior),
    each containing:
        hu          : float32 ndarray (H, W)
        z_position  : float — z from ImagePositionPatient[2] (mm), or slice index
        sop_uid     : str — SOPInstanceUID for SR references
        file_path   : str
    """
    import pydicom
    import warnings

    series_folder = Path(series_folder)
    slices = []

    for dcm_file in sorted(series_folder.rglob("*.dcm")):
        hu_array, image_id, is_valid = read_dicom_hu(str(dcm_file))
        if not is_valid or hu_array is None:
            continue

        # Read header again (lightweight) for position and UID
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            z_pos = float(getattr(ds, "ImagePositionPatient", [0, 0, 0])[2])
            sop_uid = str(getattr(ds, "SOPInstanceUID", ""))
        except Exception:
            z_pos   = len(slices)   # fall back to file order
            sop_uid = ""

        slices.append({
            "hu":        hu_array,
            "z_position": z_pos,
            "sop_uid":   sop_uid,
            "file_path": str(dcm_file),
        })

    if not slices:
        return slices

    # Sort by z-position: most superior (highest z) first for head CT
    slices.sort(key=lambda s: s["z_position"], reverse=True)
    return slices


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_slice(hu: np.ndarray) -> torch.Tensor:
    """
    HU float32 (H, W) → float32 tensor (1, 1, 224, 224) ready for the model.
    Applies medium window then resizes to 224×224.
    """
    # Medium HU window → float16 [0, 1]
    normed = apply_window(hu, WINDOW_MEDIUM).astype(np.float32)

    # Resize to 224×224 if needed
    h, w = normed.shape
    if h != TARGET_SIZE or w != TARGET_SIZE:
        t      = torch.from_numpy(normed).unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        normed = F.interpolate(
            t, size=(TARGET_SIZE, TARGET_SIZE),
            mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0).numpy()

    # (H, W) → (1, H, W) channel dim
    return torch.from_numpy(normed[np.newaxis])   # (1, 224, 224)


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    series_folder:   str,
    checkpoint_path: str         = DEFAULT_CKPT,
    device:          Optional[torch.device] = None,
    batch_size:      int         = BATCH_SIZE,
    verbose:         bool        = True,
) -> dict:
    """
    Run MaxViT ICH inference on a DICOM series folder.

    Returns a dict suitable for the ich_agent.py tool response:
    {
        series_folder   : str
        checkpoint      : str
        slice_count     : int
        valid_slices    : int
        study_level     : { class_name: { prob, positive } }
        hot_slices      : [ { slice_index, slice_z_mm, sop_uid,
                               dominant_class, prob } ]
        overall_positive: bool
        dominant_class  : str   (display name of highest-prob positive class)
        total_slices    : int
        per_slice_probs : [ { slice_index, z_mm, sop_uid, probs:{class:prob} } ]
    }
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    if verbose:
        print(f"\nICH Inference")
        print(f"  Series  : {series_folder}")
        print(f"  Device  : {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.isfile(checkpoint_path):
        return {"error": f"Checkpoint not found: {checkpoint_path}"}

    model = load_model(checkpoint_path, device)

    # ── Load slices ───────────────────────────────────────────────────────────
    if verbose:
        print(f"  Reading DICOM slices from {series_folder}...")

    slices = load_series_slices(series_folder)
    if not slices:
        return {"error": f"No valid DICOM slices found in {series_folder}"}

    if verbose:
        print(f"  Valid slices: {len(slices)}")

    # ── Batch inference ───────────────────────────────────────────────────────
    all_probs = []   # list of (6,) float32 arrays

    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    with torch.no_grad():
        for batch_start in range(0, len(slices), batch_size):
            batch_slices = slices[batch_start : batch_start + batch_size]
            batch_tensors = torch.stack(
                [preprocess_slice(s["hu"]) for s in batch_slices]
            ).to(device)   # (B, 1, 224, 224)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"),
                                    dtype=amp_dtype):
                logits = model(batch_tensors)           # (B, 6)
                probs  = torch.sigmoid(logits)          # (B, 6)

            all_probs.extend(probs.cpu().float().numpy())

            if verbose:
                done = min(batch_start + batch_size, len(slices))
                print(f"  Processed {done}/{len(slices)} slices", end="\r")

    if verbose:
        print()

    all_probs = np.array(all_probs)   # (N_slices, 6)

    # ── Study-level aggregation (max pooling across slices) ───────────────────
    study_max = all_probs.max(axis=0)   # (6,)

    study_level = {}
    for i, col in enumerate(LABEL_COLS):
        prob = float(study_max[i])
        study_level[col] = {
            "prob":     round(prob, 4),
            "positive": prob >= POSITIVE_THRESH,
            "display":  CLASS_DISPLAY.get(col, col),
        }

    # ── Hot slices (top activations per class, deduplicated) ─────────────────
    hot_slice_set: dict[int, dict] = {}

    for i, col in enumerate(LABEL_COLS):
        if not study_level[col]["positive"]:
            continue
        # Top 3 slices for this class
        top_indices = np.argsort(all_probs[:, i])[::-1][:3]
        for idx in top_indices:
            prob = float(all_probs[idx, i])
            if prob < POSITIVE_THRESH:
                break
            if idx not in hot_slice_set or prob > hot_slice_set[idx]["prob"]:
                hot_slice_set[idx] = {
                    "slice_index":    int(idx),
                    "slice_z_mm":     round(slices[idx]["z_position"], 1),
                    "sop_uid":        slices[idx]["sop_uid"],
                    "dominant_class": col,
                    "display_class":  CLASS_DISPLAY.get(col, col),
                    "prob":           round(prob, 4),
                }

    hot_slices = sorted(hot_slice_set.values(), key=lambda s: s["slice_index"])

    # ── Overall determination ─────────────────────────────────────────────────
    # Use "any" class as primary signal; fall back to max of subtypes
    overall_positive = study_level["any"]["positive"]

    # Dominant subtype: highest-prob positive class excluding "any"
    subtypes = [col for col in LABEL_COLS if col != "any"]
    pos_subtypes = [
        (col, study_level[col]["prob"])
        for col in subtypes
        if study_level[col]["positive"]
    ]
    if pos_subtypes:
        dominant_col = max(pos_subtypes, key=lambda x: x[1])[0]
        dominant_display = CLASS_DISPLAY.get(dominant_col, dominant_col)
    elif overall_positive:
        dominant_display = "intracranial hemorrhage (subtype undetermined)"
    else:
        dominant_display = ""

    # ── Per-slice probability table ───────────────────────────────────────────
    per_slice = [
        {
            "slice_index": i,
            "z_mm":        round(slices[i]["z_position"], 1),
            "sop_uid":     slices[i]["sop_uid"],
            "probs":       {
                col: round(float(all_probs[i, j]), 4)
                for j, col in enumerate(LABEL_COLS)
            },
        }
        for i in range(len(slices))
    ]

    result = {
        "series_folder":    series_folder,
        "checkpoint":       checkpoint_path,
        "slice_count":      len(slices),
        "valid_slices":     len(slices),
        "study_level":      study_level,
        "hot_slices":       hot_slices,
        "overall_positive": overall_positive,
        "dominant_class":   dominant_display,
        "total_slices":     len(slices),
        "per_slice_probs":  per_slice,
    }

    if verbose:
        _print_summary(result)

    return result


# ── Console summary ───────────────────────────────────────────────────────────

def _print_summary(result: dict):
    print(f"\n  {'─'*52}")
    print(f"  Study result : "
          f"{'POSITIVE — ' + result['dominant_class'] if result['overall_positive'] else 'NEGATIVE'}")
    print(f"  {'─'*52}")
    print(f"  {'Class':<26} {'Prob':>6}  {'Call':>8}")
    print(f"  {'─'*52}")
    for col, vals in result["study_level"].items():
        call = "POSITIVE" if vals["positive"] else "negative"
        print(f"  {CLASS_DISPLAY.get(col, col):<26} {vals['prob']:>6.3f}  {call:>8}")
    print(f"  {'─'*52}")

    if result["hot_slices"]:
        print(f"\n  Hot slices:")
        for s in result["hot_slices"]:
            print(f"    Slice {s['slice_index']:3d} "
                  f"(z={s['slice_z_mm']:+.1f} mm)  "
                  f"{CLASS_DISPLAY.get(s['dominant_class'], s['dominant_class'])}  "
                  f"score={s['prob']*100:.1f}%")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run MaxViT ICH inference on a DICOM series folder"
    )
    parser.add_argument("series_folder",
                        help="Path to the DICOM series folder")
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT,
                        help="Path to best_maxvit_ich.pth checkpoint")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--output-json", default=None,
                        help="Optional path to write results as JSON")
    args = parser.parse_args()

    result = run_inference(
        series_folder   = args.series_folder,
        checkpoint_path = args.checkpoint,
        batch_size      = args.batch_size,
    )

    if args.output_json:
        # per_slice_probs can be large — omit from saved JSON if not needed
        out = {k: v for k, v in result.items() if k != "per_slice_probs"}
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nResults written to {args.output_json}")


if __name__ == "__main__":
    main()
