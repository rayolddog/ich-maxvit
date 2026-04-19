# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

NewICH is an end-to-end intracranial hemorrhage (ICH) AI detection pipeline. It combines a MaxViT deep learning classifier (trained on RSNA ICH data), calibrated CT preprocessing using fixed Hounsfield Unit (HU) windows, optional Claude-based agentic orchestration, and a browser-based radiology worklist viewer with DICOM Structured Report generation. It detects 5 ICH subtypes: epidural, intraparenchymal, intraventricular, subarachnoid, subdural.

## Setup

```bash
pip install -r requirements.txt
python download_checkpoint.py   # Downloads best_maxvit_ich.pth (~1.4 GB) from HuggingFace Hub
```

## Running

```bash
# Direct mode (no API key, ~16s on GPU)
python run_demo_direct.py

# Agentic mode (requires ANTHROPIC_API_KEY, Claude writes reports)
export ANTHROPIC_API_KEY="sk-ant-..."
python run_demo_agent.py

# Web worklist (use Firefox or Edge — Chrome blocks localhost:5050)
python ich_worklist.py
```

## Inference on Custom Data

```bash
python ich_inference.py /path/to/dicom/series \
    --checkpoint checkpoints_maxvit/best_maxvit_ich.pth \
    --output-json results.json
```

## Evaluation & Training

```bash
# Evaluate model on held-out test set → checkpoints_maxvit/test_metrics.json
python evaluate_maxvit_test.py

# Train from scratch (requires prebuilt .npz cache)
python train_maxvit.py \
    --cache-dir /data/cache_medhu_train \
    --test-dir /data/cache_medhu_test \
    --labels-csv /data/stage_2_train.csv \
    --splits-file /data/data_splits.json \
    --batch-size 256 --workers 16 --epochs 50
```

No automated test suite — validation is done via `evaluate_maxvit_test.py` and visual inspection of the browser worklist with demo studies.

## Architecture

The pipeline runs in 5 steps:

```
DICOM folder → scan_study() → series selection → run_ich_inference() → generate_dicom_sr() → flag_worklist()
```

### HU Window Preprocessing (`hu_windows.py`)
The core architectural innovation. CT pixels are physical measurements (Hounsfield Units), so per-image normalization destroys absolute density information. Instead, three fixed global windows are used:
- **NARROW** (48–90 HU): acute ICH zone, maximum blood/brain contrast — used for inference
- **MEDIUM** (−200 to 200): brain anatomy context
- **WIDE** (−1024 to 3071): full CT range

This ensures 65 HU blood always maps to the same tensor value regardless of surrounding anatomy.

### MaxViT Model (`ich_inference.py`, `train_maxvit.py`)
- MaxViT-Base (timm), 118.7M params, 1-channel 224×224 input
- 6-class multi-label output (5 subtypes + "any ICH") with sigmoid (not softmax) — a slice can have multiple hemorrhage types
- Focal loss (γ=2.0) to handle class imbalance; BF16 precision
- Study-level aggregation via **max pooling** across slices: the highest per-slice probability per class becomes the study score
- Hot slices = top 3 per positive class, deduplicated — these are what the radiologist sees first in the viewer

### DICOM Structured Report (`ich_dicom_sr.py`)
Generates a separate DICOM SR object (SOP 1.2.840.10008.5.1.4.1.1.88.33) — not an overlay on source images. Encodes probabilities, IMAGE references to hot slices, Bayesian metrics (PPV/NPV/LR+/LR− at configurable prevalence), and a confusion matrix for 1,000 cases. Loads `test_metrics.json` at import time; falls back to hardcoded defaults if missing.

### Flask Worklist (`ich_worklist.py`)
Single-page app with persistent `worklist.json` (thread-safe with `_lock`). Serves real DICOM PNGs or synthetic brain renderings for studies without DICOM files. Priority tiers: RED (stat), ORANGE (AI-positive), WHITE (negative). HU Overlay highlights pixels in the 48–90 HU acute ICH range in red.

### Agentic Orchestration (`ich_agent.py`)
Optional layer using Claude Opus (tool-use API). Tools: `scan_study`, `run_ich_inference`, `generate_dicom_sr`, `flag_worklist`. Claude selects the correct series from metadata, writes natural-language impressions, and looks up local prevalence from the SQLite DB (`prevalence_db.py`) to compute patient-specific PPV/NPV. Direct mode bypasses the agent and uses templated reports.

### DICOM Reader (`dicom_reader_1ch.py`)
Converts DICOM pixel arrays to HU using RescaleSlope/RescaleIntercept, with plausibility validation to catch uncalibrated scouts and bad UID conversions.

### Prevalence Database (`prevalence_db.py`)
SQLite DB tracking study outcomes by location and time period. Used by the agent to retrieve location-specific prevalence (ED vs. ICU vs. neuro ward) for PPV/NPV recalculation.

## Key Design Decisions

- **Fixed HU windows, not per-image normalization**: Validated to give 5.5× better blood/gray matter separation.
- **"Any ICH" as a 6th class**: Trained independently from the 5 subtypes for a separate "was there any ICH" confidence score.
- **Youden J threshold selection**: Maximizes sensitivity + specificity (not AUC). Clinically appropriate for a screening test where missed ICH is catastrophic.
- **Per-slice probs dropped before agent context**: `run_inference()` pops the large `per_slice_probs` array before returning to Claude to avoid bloating message history; the data is still available to the worklist.
- **Claude for orchestration, not diagnosis**: Inference is pure PyTorch — deterministic, offline, auditable. Claude writes reports and selects series only.
- **No PACS integration**: `flag_worklist()` writes to `worklist.json`. In production, replace with PACS API calls.
