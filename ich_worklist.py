#!/usr/bin/env python3
"""
ICH AI Worklist — Flask web server.

Maintains a JSON worklist file updated by the ICH agent when studies
are processed.  Serves a single-page UI showing studies with priority
flags, AI results, and report text.

Priority flag tiers (visually distinct):
  RED    — Physician-ordered STAT (existing workflow, simulated here)
  ORANGE — AI-detected ICH (new tier, not physician-ordered)
  WHITE  — AI screened, no ICH detected

Usage:
    python ich_worklist.py                        # start server (port 5050)
    python ich_worklist.py --port 8888
    python ich_worklist.py --add-demo             # seed with demo studies
"""

import os
import sys
import json
import time
import uuid
import argparse
import threading
from pathlib import Path
from datetime import datetime

from flask import Flask, jsonify, request, render_template_string, Response

try:
    from PIL import Image, ImageDraw, ImageFilter
    import numpy as np
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

# ── Worklist store ─────────────────────────────────────────────────────────────

WORKLIST_PATH = Path(__file__).parent / "worklist.json"

_lock = threading.Lock()


def _load() -> list:
    if WORKLIST_PATH.exists():
        with open(WORKLIST_PATH) as f:
            return json.load(f)
    return []


def _save(studies: list):
    with open(WORKLIST_PATH, "w") as f:
        json.dump(studies, f, indent=2)


def add_study(study: dict) -> dict:
    """
    Add or update a study in the worklist.
    study dict keys:
        study_uid       str   — DICOM StudyInstanceUID
        patient_id      str
        patient_name    str   (optional)
        indication      str
        exam_time       str   — ISO timestamp of exam acquisition
        modality        str   — e.g. "CT"
        description     str   — study/series description
        physician_stat  bool  — True = physician-ordered STAT (red)
        ai_positive     bool  — True = AI detected ICH (orange)
        ai_result       dict  — study_level probs, dominant_class, hot_slices
        report_paragraph str  — body paragraph from agent
        impression_bullet str — impression bullet from agent
        sr_path         str   — path to DICOM SR file
        processed_time  str   — ISO timestamp when AI processed
        prevalence      float
        metrics         dict  — ppv, npv, pain_index, confusion_matrix
    """
    with _lock:
        studies = _load()

        # Update existing entry if study_uid matches
        for i, s in enumerate(studies):
            if s.get("study_uid") == study.get("study_uid"):
                studies[i] = {**s, **study,
                               "processed_time": datetime.now().isoformat()}
                _save(studies)
                return studies[i]

        # New entry
        study.setdefault("id",             str(uuid.uuid4())[:8])
        study.setdefault("exam_time",      datetime.now().isoformat())
        study.setdefault("processed_time", datetime.now().isoformat())
        study.setdefault("physician_stat", False)
        study.setdefault("ai_positive",    False)
        studies.append(study)

        # Sort: physician STAT first, then AI positive, then by exam time
        studies.sort(key=lambda s: (
            0 if s.get("physician_stat") else
            1 if s.get("ai_positive")    else 2,
            s.get("exam_time", "")
        ))

        _save(studies)
        return study


def get_studies() -> list:
    with _lock:
        return _load()


# ── Public API used by ich_agent.py ──────────────────────────────────────────

def record_result(
    study_uid:         str,
    patient_id:        str,
    indication:        str,
    ai_positive:       bool,
    ai_result:         dict,
    report_paragraph:  str  = "",
    impression_bullet: str  = "",
    sr_path:           str  = "",
    metrics:           dict = None,
    prevalence:        float = 0.02,
    physician_stat:    bool  = False,
    description:       str  = "",
    patient_name:      str  = "",
) -> dict:
    """Called by ich_agent._flag_worklist() to persist a study result."""
    dominant = ai_result.get("dominant_class", "") if ai_positive else ""
    return add_study({
        "study_uid":         study_uid,
        "patient_id":        patient_id,
        "patient_name":      patient_name or patient_id,
        "indication":        indication,
        "modality":          "CT",
        "description":       description or "CT HEAD",
        "physician_stat":    physician_stat,
        "ai_positive":       ai_positive,
        "dominant_class":    dominant,
        "ai_result":         ai_result,
        "report_paragraph":  report_paragraph,
        "impression_bullet": impression_bullet,
        "sr_path":           sr_path,
        "metrics":           metrics or {},
        "prevalence":        prevalence,
    })


# ── Demo seed data ────────────────────────────────────────────────────────────

DEMO_STUDIES = [
    {
        "study_uid":        "1.2.840.99999.1.1001",
        "patient_id":       "PT-1001",
        "patient_name":     "SMITH^JOHN",
        "indication":       "Headache, vomiting",
        "modality":         "CT",
        "description":      "CT HEAD W/O CONTRAST",
        "physician_stat":   False,
        "ai_positive":      True,
        "dominant_class":   "subdural hematoma",
        "ai_result": {
            "overall_positive": True,
            "dominant_class":   "subdural hematoma",
            "study_level": {
                "epidural":         {"prob": 0.03, "positive": False},
                "intraparenchymal": {"prob": 0.06, "positive": False},
                "intraventricular": {"prob": 0.04, "positive": False},
                "subarachnoid":     {"prob": 0.05, "positive": False},
                "subdural":         {"prob": 0.87, "positive": True},
                "any":              {"prob": 0.91, "positive": True},
            },
            "hot_slices": [
                {"slice_index": 19, "slice_z_mm": 14.5,
                 "dominant_class": "subdural", "prob": 0.91},
                {"slice_index": 21, "slice_z_mm": 10.2,
                 "dominant_class": "subdural", "prob": 0.95},
            ],
            "valid_slices": 42,
        },
        "report_paragraph": (
            "ARTIFICIAL INTELLIGENCE SCREENING — INTRACRANIAL HEMORRHAGE: "
            "The noncontrast axial CT head sequence (42 slices) was evaluated "
            "by an AI model (MaxViT-Base, mean AUC 0.988) trained for detection "
            "of intracranial hemorrhage across six categories. "
            "Hemorrhagic findings were identified. The model detected findings "
            "most consistent with subdural hematoma (model score 87%) on slices 19–21 "
            "(z = +10 to +15 mm). Intraparenchymal (6%), epidural (3%), "
            "and intraventricular (4%) subtypes were not favored. "
            "PPV 40.8% at assumed prevalence of 2% (headache). "
            "This study has been flagged for priority AI-detected review. "
            "AI-assisted screening. Not FDA-cleared. Requires radiologist confirmation."
        ),
        "impression_bullet": (
            "• AI SCREENING (ICH): POSITIVE — subdural hematoma, slices 19–21. "
            "Study flagged for priority AI-detected radiologist review."
        ),
        "sr_path":    "",
        "metrics":    {"ppv": 0.408, "npv": 1.0, "pain_index": 1.45,
                       "tp": 20, "fn": 0, "fp": 29, "tn": 951},
        "prevalence": 0.02,
    },
    {
        "study_uid":        "1.2.840.99999.1.1002",
        "patient_id":       "PT-1002",
        "patient_name":     "JONES^MARY",
        "indication":       "Fall, altered mental status",
        "modality":         "CT",
        "description":      "CT HEAD W/O CONTRAST",
        "physician_stat":   True,
        "ai_positive":      False,
        "dominant_class":   "",
        "ai_result": {
            "overall_positive": False,
            "dominant_class":   "",
            "study_level": {
                "epidural":         {"prob": 0.02, "positive": False},
                "intraparenchymal": {"prob": 0.03, "positive": False},
                "intraventricular": {"prob": 0.02, "positive": False},
                "subarachnoid":     {"prob": 0.04, "positive": False},
                "subdural":         {"prob": 0.06, "positive": False},
                "any":              {"prob": 0.07, "positive": False},
            },
            "hot_slices":   [],
            "valid_slices": 38,
        },
        "report_paragraph": (
            "ARTIFICIAL INTELLIGENCE SCREENING — INTRACRANIAL HEMORRHAGE: "
            "The noncontrast axial CT head sequence (38 slices) was evaluated "
            "by an AI model (MaxViT-Base, mean AUC 0.988). "
            "No hemorrhagic findings were identified. "
            "NPV 99.9% at assumed prevalence of 8% (altered mental status). "
            "AI-assisted screening. Not FDA-cleared. Requires radiologist confirmation."
        ),
        "impression_bullet": (
            "• AI SCREENING (ICH): NEGATIVE. NPV 99.9% at 8% assumed prevalence. "
            "Routine radiologist review."
        ),
        "sr_path":    "",
        "metrics":    {"ppv": 0.74, "npv": 0.999, "pain_index": 0.35,
                       "tp": 78, "fn": 2, "fp": 27, "tn": 893},
        "prevalence": 0.08,
    },
    {
        "study_uid":        "1.2.840.99999.1.1003",
        "patient_id":       "PT-1003",
        "patient_name":     "WILLIAMS^ROBERT",
        "indication":       "Post-op day 1, craniotomy",
        "modality":         "CT",
        "description":      "CT HEAD W/O CONTRAST",
        "physician_stat":   False,
        "ai_positive":      True,
        "dominant_class":   "intraparenchymal hemorrhage",
        "ai_result": {
            "overall_positive": True,
            "dominant_class":   "intraparenchymal hemorrhage",
            "study_level": {
                "epidural":         {"prob": 0.11, "positive": False},
                "intraparenchymal": {"prob": 0.82, "positive": True},
                "intraventricular": {"prob": 0.34, "positive": False},
                "subarachnoid":     {"prob": 0.21, "positive": False},
                "subdural":         {"prob": 0.19, "positive": False},
                "any":              {"prob": 0.89, "positive": True},
            },
            "hot_slices": [
                {"slice_index": 14, "slice_z_mm": 22.0,
                 "dominant_class": "intraparenchymal", "prob": 0.82},
            ],
            "valid_slices": 44,
        },
        "report_paragraph": (
            "ARTIFICIAL INTELLIGENCE SCREENING — INTRACRANIAL HEMORRHAGE: "
            "The noncontrast axial CT head sequence (44 slices) was evaluated "
            "by an AI model (MaxViT-Base, mean AUC 0.988). "
            "Hemorrhagic findings were identified. The model detected findings "
            "most consistent with intraparenchymal hemorrhage (model score 82%) on slice 14 "
            "(z = +22 mm). "
            "PPV 78.1% at assumed prevalence of 20% (post-op craniotomy). "
            "This study has been flagged for priority AI-detected review. "
            "AI-assisted screening. Not FDA-cleared. Requires radiologist confirmation."
        ),
        "impression_bullet": (
            "• AI SCREENING (ICH): POSITIVE — intraparenchymal hemorrhage, slice 14. "
            "Study flagged for priority AI-detected radiologist review."
        ),
        "sr_path":    "",
        "metrics":    {"ppv": 0.781, "npv": 0.996, "pain_index": 0.28,
                       "tp": 196, "fn": 4, "fp": 55, "tn": 745},
        "prevalence": 0.20,
    },
]


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Radiology AI Worklist — ICH Screening</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #1a1a2e;
      color: #e0e0e0;
      min-height: 100vh;
    }

    header {
      background: #16213e;
      border-bottom: 2px solid #0f3460;
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }

    header h1 {
      font-size: 1.2rem;
      font-weight: 600;
      letter-spacing: 0.04em;
      color: #a8d8ea;
    }

    .legend {
      display: flex;
      gap: 18px;
      font-size: 0.78rem;
      align-items: center;
    }

    .legend-item {
      display: flex;
      align-items: center;
      gap: 6px;
    }

    .dot {
      width: 12px; height: 12px;
      border-radius: 50%;
      flex-shrink: 0;
    }

    .dot-stat    { background: #e53935; }
    .dot-ai      { background: #ff7043; }
    .dot-normal  { background: #546e7a; }

    .refresh-info {
      font-size: 0.72rem;
      color: #607d8b;
      padding: 6px 24px;
      background: #16213e;
    }

    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }

    thead th {
      background: #0f3460;
      color: #90caf9;
      padding: 10px 14px;
      text-align: left;
      font-weight: 600;
      letter-spacing: 0.03em;
      position: sticky;
      top: 0;
      z-index: 10;
    }

    tbody tr {
      border-bottom: 1px solid #1e2d40;
      cursor: pointer;
      transition: background 0.15s;
    }

    tbody tr:hover { background: #1e2d40; }

    tbody tr.stat-row     { border-left: 4px solid #e53935; }
    tbody tr.ai-pos-row   { border-left: 4px solid #ff7043; }
    tbody tr.normal-row   { border-left: 4px solid #37474f; }

    td {
      padding: 10px 14px;
      vertical-align: middle;
    }

    .badge {
      display: inline-block;
      padding: 3px 9px;
      border-radius: 12px;
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.05em;
    }

    .badge-stat   { background: #b71c1c; color: #fff; }
    .badge-ai     { background: #bf360c; color: #fff; }
    .badge-neg    { background: #263238; color: #90a4ae; border: 1px solid #37474f; }

    .prob-pos { color: #ef5350; font-weight: 700; }
    .prob-neg { color: #78909c; }

    /* Modal */
    /* ── Viewer modal ── */
    .viewer-overlay {
      display: none; position: fixed; inset: 0;
      background: rgba(0,0,0,0.92); z-index: 200;
      align-items: center; justify-content: center;
    }
    .viewer-overlay.open { display: flex; }
    .viewer {
      background: #0d0d1a; border: 1px solid #1e3a5f;
      border-radius: 8px; width: 96vw; height: 92vh;
      display: flex; flex-direction: column; overflow: hidden;
    }
    .viewer-header {
      background: #16213e; padding: 10px 18px;
      display: flex; align-items: center; justify-content: space-between;
      border-bottom: 1px solid #1e3a5f; flex-shrink: 0;
    }
    .viewer-header h2 { font-size: 1rem; color: #90caf9; font-weight: 600; }
    .viewer-close {
      background: none; border: none; color: #aaa; font-size: 1.3rem;
      cursor: pointer; padding: 4px 8px; border-radius: 4px;
    }
    .viewer-close:hover { color: #fff; background: #c0392b; }
    .viewer-body {
      display: flex; flex: 1; overflow: hidden;
    }
    /* Left — image panel */
    .viewer-image-panel {
      flex: 0 0 52%; display: flex; flex-direction: column;
      align-items: center; justify-content: center;
      background: #000; border-right: 1px solid #1e3a5f; padding: 12px;
    }
    .viewer-img-wrap { position: relative; }
    .viewer-img-wrap img {
      display: block; max-width: 100%; max-height: calc(92vh - 160px);
      border: 1px solid #263238;
    }
    .slice-nav {
      display: flex; align-items: center; gap: 16px; margin-top: 10px;
    }
    .nav-btn {
      background: #1e3a5f; border: none; color: #90caf9;
      font-size: 1.5rem; width: 40px; height: 40px; border-radius: 50%;
      cursor: pointer; display: flex; align-items: center; justify-content: center;
    }
    .nav-btn:hover { background: #0f3460; }
    .nav-btn:disabled { opacity: 0.3; cursor: default; }
    .slice-info { color: #90caf9; font-size: 0.82rem; text-align: center; }
    .demo-note { color: #c0392b; font-size: 0.72rem; margin-top: 4px; }
    /* Right — SR panel */
    .viewer-sr-panel {
      flex: 1; overflow-y: auto; padding: 18px 20px;
      display: flex; flex-direction: column; gap: 16px;
    }
    .sr-section h3 {
      font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em;
      color: #546e7a; margin-bottom: 8px; border-bottom: 1px solid #1e3a5f;
      padding-bottom: 4px;
    }
    .sr-impression {
      background: #0f2a1a; border-left: 3px solid #4caf50;
      padding: 10px 14px; border-radius: 4px;
      color: #a5d6a7; font-size: 0.9rem; line-height: 1.5;
    }
    .sr-impression.positive {
      background: #2a1a0f; border-color: #ff7043; color: #ffab91;
    }
    .prob-bar-row {
      display: flex; align-items: center; gap: 8px; margin-bottom: 5px;
    }
    .prob-bar-label { width: 130px; font-size: 0.8rem; color: #b0bec5; flex-shrink: 0; }
    .prob-bar-track {
      flex: 1; background: #1a2a3a; border-radius: 3px; height: 14px; overflow: hidden;
    }
    .prob-bar-fill {
      height: 100%; background: #37474f; border-radius: 3px;
      transition: width 0.3s ease;
    }
    .prob-bar-fill.positive { background: #e53935; }
    .prob-bar-val { width: 42px; font-size: 0.8rem; color: #90caf9; text-align: right; }
    .sr-metrics-grid {
      display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
    }
    .sr-metric { background: #0d1b2a; border-radius: 4px; padding: 8px 10px; }
    .sr-metric .label { font-size: 0.68rem; color: #546e7a; text-transform: uppercase; }
    .sr-metric .value { font-size: 1.0rem; font-weight: 700; color: #90caf9; }
    .sr-metric .value.pain  { color: #ff7043; }
    .sr-report {
      font-size: 0.82rem; color: #b0bec5; line-height: 1.6;
      background: #0a0f1a; padding: 10px 12px; border-radius: 4px;
    }
    .sr-disclaimer { font-size: 0.72rem; color: #37474f; font-style: italic; }
    .name-link {
      color: #90caf9; cursor: pointer; text-decoration: underline dotted;
    }
    .name-link:hover { color: #fff; }

    .modal-overlay {
      display: none;
      position: fixed; inset: 0;
      background: rgba(0,0,0,0.7);
      z-index: 100;
      justify-content: center;
      align-items: flex-start;
      padding: 40px 20px;
      overflow-y: auto;
    }

    .modal-overlay.open { display: flex; }

    .modal {
      background: #16213e;
      border: 1px solid #0f3460;
      border-radius: 8px;
      width: 100%;
      max-width: 780px;
      padding: 28px;
      position: relative;
    }

    .modal-close {
      position: absolute;
      top: 14px; right: 18px;
      background: none;
      border: none;
      color: #90a4ae;
      font-size: 1.4rem;
      cursor: pointer;
    }

    .modal h2 {
      font-size: 1rem;
      color: #a8d8ea;
      margin-bottom: 18px;
      padding-bottom: 10px;
      border-bottom: 1px solid #0f3460;
    }

    .modal-section { margin-bottom: 20px; }

    .modal-section h3 {
      font-size: 0.75rem;
      color: #607d8b;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }

    .report-text {
      background: #0d1b2a;
      border: 1px solid #1e3a5f;
      border-radius: 4px;
      padding: 14px;
      font-size: 0.83rem;
      line-height: 1.6;
      white-space: pre-wrap;
      color: #cfd8dc;
    }

    .impression-text {
      background: #0d1b2a;
      border-left: 3px solid #ff7043;
      padding: 12px 14px;
      font-size: 0.85rem;
      font-weight: 600;
      color: #ffccbc;
    }

    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 10px;
    }

    .metric-card {
      background: #0d1b2a;
      border: 1px solid #1e3a5f;
      border-radius: 4px;
      padding: 10px 12px;
      text-align: center;
    }

    .metric-card .label {
      font-size: 0.68rem;
      color: #607d8b;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 4px;
    }

    .metric-card .value {
      font-size: 1.15rem;
      font-weight: 700;
      color: #90caf9;
    }

    .metric-card .value.pain { color: #ff7043; }
    .metric-card .value.good { color: #66bb6a; }

    .confusion-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
    }

    .confusion-table th, .confusion-table td {
      border: 1px solid #1e3a5f;
      padding: 8px 12px;
      text-align: center;
    }

    .confusion-table th { background: #0f3460; color: #90caf9; }
    .confusion-table .tp { color: #66bb6a; font-weight: 700; }
    .confusion-table .fp { color: #ef5350; font-weight: 700; }
    .confusion-table .fn { color: #ffa726; font-weight: 700; }
    .confusion-table .tn { color: #78909c; }

    .class-probs {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .class-prob-item {
      background: #0d1b2a;
      border: 1px solid #1e3a5f;
      border-radius: 4px;
      padding: 6px 10px;
      font-size: 0.78rem;
    }

    .class-prob-item.positive {
      border-color: #ff7043;
      color: #ffccbc;
    }

    .empty-state {
      text-align: center;
      padding: 60px;
      color: #546e7a;
    }
  </style>
</head>
<body>

<header>
  <h1>&#128203; Radiology AI Worklist — ICH Screening</h1>
  <div class="legend">
    <div class="legend-item">
      <div class="dot dot-stat"></div> Physician STAT
    </div>
    <div class="legend-item">
      <div class="dot dot-ai"></div> AI-Detected ICH
    </div>
    <div class="legend-item">
      <div class="dot dot-normal"></div> AI Negative
    </div>
  </div>
</header>

<div class="refresh-info">Auto-refreshes every 15 seconds &nbsp;|&nbsp;
  <span id="last-update">Loading...</span>
</div>

<table>
  <thead>
    <tr>
      <th>Priority</th>
      <th>Patient</th>
      <th>Indication</th>
      <th>Description</th>
      <th>AI Finding</th>
      <th>p(any ICH)</th>
      <th>PPV</th>
      <th title="False positive overrides per true detection (FP:TP ratio). Number of incorrect AI positive calls the radiologist must review and reject for each true ICH found.">Pain Index ⓘ</th>
      <th>Processed</th>
    </tr>
  </thead>
  <tbody id="worklist-body">
    <tr><td colspan="9" class="empty-state">Loading worklist...</td></tr>
  </tbody>
</table>

<!-- Slice Viewer -->
<div class="viewer-overlay" id="viewer">
  <div class="viewer">
    <div class="viewer-header">
      <h2 id="viewer-title">Slice Viewer</h2>
      <button class="viewer-close" onclick="closeViewer()">&#10005;</button>
    </div>
    <div class="viewer-body">

      <!-- Left: image + navigation -->
      <div class="viewer-image-panel">
        <div class="viewer-img-wrap">
          <img id="viewer-img" src="" alt="CT slice" />
        </div>
        <div class="slice-nav">
          <button class="nav-btn" id="nav-prev" onclick="navigateSlice(-1)">&#8592;</button>
          <div class="slice-info" id="viewer-slice-info">—</div>
          <button class="nav-btn" id="nav-next" onclick="navigateSlice(+1)">&#8594;</button>
        </div>
        <div style="text-align:center;margin-top:8px">
          <button id="hu-overlay-btn" onclick="toggleHuOverlay()"
            style="padding:5px 16px;font-size:0.78rem;font-weight:600;
                   background:#b71c1c;color:#fff;border:none;border-radius:5px;
                   cursor:pointer;letter-spacing:0.03em;transition:background 0.15s"
            title="Toggle red pixel overlay showing tissue density in the acute ICH Hounsfield unit range (48–90 HU)">
            HU Overlay
          </button>
        </div>
        <div id="hu-overlay-caption" style="display:none;margin-top:6px;padding:8px 12px;
             background:#311313;border-radius:4px;font-size:0.72rem;
             color:#ef9a9a;line-height:1.6">
          <b>HU Overlay — purpose and limitations</b><br>
          Red pixels mark tissue density 48–90 HU, the range of <i>acute</i>
          intracranial hemorrhage. This is a physics-based display aid computed
          directly from raw DICOM pixel values. <b>It is not an activation map
          of the neural network</b> and does not show where the AI detected
          hemorrhage.<br><br>
          <b>What the overlay will not highlight:</b> Chronic subdural hematomas
          (typically ≥ 1 week old) become isodense or hypodense relative to brain
          and fall below this HU range. A positive AI result for chronic subdural
          may show little or no red overlay — this is expected and does not
          invalidate the AI finding.<br><br>
          <b>Expected normal findings in this range:</b> partial volume averaging
          at the inner table of the skull (bright ring at brain edge), physiological
          calcifications (choroid plexus, pineal, basal ganglia), metal implant
          artifacts, and beam-hardening streaks near dense bone. These are
          normal and should be mentally subtracted.<br><br>
          The signal of clinical interest is <b>unexpected red pixels in the
          parenchyma or extra-axial space</b>, away from bone edges and known
          calcification sites. The overlay is intended to focus radiologist
          attention, not to replace it.
        </div>
        <div class="demo-note" id="viewer-demo-note"></div>
      </div>

      <!-- Right: SR content -->
      <div class="viewer-sr-panel">

        <div class="sr-section">
          <h3>Impression</h3>
          <div class="sr-impression" id="sr-impression"></div>
        </div>

        <div class="sr-section">
          <h3>AI Probability by Class</h3>
          <div id="sr-probs"></div>
        </div>

        <div class="sr-section">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <h3 style="margin:0">Performance at Assumed Prevalence</h3>
            <button onclick="toggleBayesExplainer()"
              style="padding:5px 14px;font-size:0.78rem;font-weight:600;
                     background:#546e7a;color:#fff;border:none;border-radius:5px;
                     cursor:pointer;letter-spacing:0.03em;transition:background 0.15s"
              title="Plain-language explanation of Bayesian statistics used in this report">
              In-Depth Explanation
            </button>
          </div>
          <div class="sr-metrics-grid" id="sr-metrics"></div>
          <div id="sr-metrics-source" style="font-size:0.7rem;color:#37474f;margin-top:6px"></div>

          <div id="bayes-explainer" style="display:none;margin-top:14px;padding:14px 16px;
               background:#eceff1;border-left:4px solid #546e7a;border-radius:4px;
               font-size:0.82rem;line-height:1.65;color:#263238">
            <strong style="font-size:0.88rem;display:block;margin-bottom:8px">
              Understanding the Statistics — A Bayesian Perspective
            </strong>
            <p style="margin:0 0 10px 0">
              The AI model is basically a sophisticated algorithm that detects structures in
              images and tests whether those structures match how acute intracranial hemorrhage
              appears on CT scans. The model is effective at determining whether a detected
              structure is likely to represent an area of hemorrhage. Because of extensive
              training on noncontrast CT images of the head, it is possible to determine how
              likely a finding represents intracranial hemorrhage.
            </p>
            <p style="margin:0 0 10px 0">
              The likelihood that a given structure represents hemorrhage is expressed by the
              Bayesian inference statistics of likelihood ratio, prevalence of disease, and
              positive and negative predictive value. The most important results are the two
              statistics: <strong>Positive Predictive Value (PPV)</strong> and
              <strong>Negative Predictive Value (NPV)</strong>.
            </p>
            <p style="margin:0 0 10px 0">
              Because of the low prevalence of intracranial hemorrhage in the general patient
              population undergoing head CT, even a powerful test will likely result in fewer
              than 30% of AI-identified cases representing true positive results. That
              interpretation of positive test results used to identify rare diseases is a
              principle we all learned in medical school.
            </p>
            <p style="margin:0">
              The <strong>Pain Index</strong> (FP:TP ratio) quantifies the practical
              consequence: for each true hemorrhage the AI correctly identifies, the
              radiologist must review and override this number of incorrect AI-positive
              flags — representing additional review time and cognitive load under the
              time pressure of clinical practice.
            </p>
          </div>
        </div>

        <div class="sr-section">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <h3 style="margin:0">Report Body Paragraph</h3>
            <button id="copy-report-btn" onclick="copyReport()"
              style="padding:5px 14px;font-size:0.78rem;font-weight:600;
                     background:#1565c0;color:#fff;border:none;border-radius:5px;
                     cursor:pointer;letter-spacing:0.03em;transition:background 0.15s"
              title="Copy paragraph to clipboard for pasting into PowerScribe or other reporting software">
              Copy
            </button>
          </div>
          <div class="sr-report" id="sr-report"></div>
        </div>

        <div class="sr-section">
          <div class="sr-disclaimer" id="sr-disclaimer"></div>
        </div>

      </div>
    </div>
  </div>
</div>

<!-- Detail Modal -->
<div class="modal-overlay" id="modal">
  <div class="modal">
    <button class="modal-close" onclick="closeModal()">&#10005;</button>
    <h2 id="modal-title">Study Detail</h2>

    <div class="modal-section" id="modal-impression-section">
      <h3>Impression Bullet</h3>
      <div class="impression-text" id="modal-impression"></div>
    </div>

    <div class="modal-section">
      <h3>Report Body Paragraph</h3>
      <div class="report-text" id="modal-report"></div>
    </div>

    <div class="modal-section">
      <h3>Per-Class Probabilities</h3>
      <div class="class-probs" id="modal-class-probs"></div>
    </div>

    <div class="modal-section">
      <h3>Performance at Assumed Prevalence</h3>
      <div class="metrics-grid" id="modal-metrics"></div>
    </div>

    <div class="modal-section">
      <h3>Confusion Matrix (per 1,000 cases)</h3>
      <table class="confusion-table" id="modal-confusion"></table>
    </div>
  </div>
</div>

<script>
const CLASS_DISPLAY = {
  epidural:         "Epidural",
  intraparenchymal: "Intraparenchymal",
  intraventricular: "Intraventricular",
  subarachnoid:     "Subarachnoid",
  subdural:         "Subdural",
  any:              "Any ICH",
};

function fmt_pct(v) {
  return (v * 100).toFixed(1) + "%";
}

function fmt_time(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
}

async function loadWorklist() {
  const resp = await fetch("/api/studies");
  const studies = await resp.json();
  const tbody = document.getElementById("worklist-body");

  document.getElementById("last-update").textContent =
    "Last update: " + new Date().toLocaleTimeString();

  if (studies.length === 0) {
    tbody.innerHTML = '<tr><td colspan="9" class="empty-state">' +
      'No studies processed yet. Run the ICH agent to populate the worklist.</td></tr>';
    return;
  }

  tbody.innerHTML = studies.map(s => {
    const rowClass = s.physician_stat ? "stat-row"
                   : s.ai_positive    ? "ai-pos-row"
                   :                    "normal-row";

    const badge = s.physician_stat
      ? '<span class="badge badge-stat">STAT</span>'
      : s.ai_positive
        ? '<span class="badge badge-ai">AI&#9651;ICH</span>'
        : '<span class="badge badge-neg">AI neg</span>';

    const finding = s.ai_positive
      ? `<span class="prob-pos">${s.dominant_class || "ICH detected"}</span>`
      : '<span class="prob-neg">No ICH</span>';

    const prob_any = s.ai_result?.study_level?.any?.prob ?? null;
    const prob_str = prob_any !== null
      ? `<span class="${prob_any >= 0.5 ? "prob-pos" : "prob-neg"}">${prob_any.toFixed(3)}</span>`
      : "—";

    const ppv = s.metrics?.ppv;
    const pi  = s.metrics?.pain_index;
    const prev_pct = s.prevalence ? (s.prevalence * 100).toFixed(0) + "%" : "—";

    const sj = JSON.stringify(JSON.stringify(s));
    return `<tr class="${rowClass}" onclick='openModal(${sj})'>
      <td>${badge}</td>
      <td><span class="name-link" onclick='event.stopPropagation();openViewer(${sj})'>${s.patient_name || s.patient_id}</span><br>
          <small style="color:#607d8b">${s.patient_id}</small></td>
      <td>${s.indication || "—"}</td>
      <td>${s.description || "—"}</td>
      <td>${finding}</td>
      <td>${prob_str}</td>
      <td>${ppv !== undefined ? fmt_pct(ppv) : "—"}
          <small style="color:#607d8b"> @${prev_pct}</small></td>
      <td>${pi !== undefined ? pi.toFixed(2) : "—"}</td>
      <td>${fmt_time(s.processed_time)}</td>
    </tr>`;
  }).join("");
}

function openModal(json_str) {
  const s = JSON.parse(json_str);

  document.getElementById("modal-title").textContent =
    (s.patient_name || s.patient_id) + " — " + (s.description || "CT HEAD");

  // Impression
  const imp = s.impression_bullet || "";
  document.getElementById("modal-impression").textContent =
    imp || "Not yet generated.";
  document.getElementById("modal-impression-section").style.display =
    imp ? "block" : "none";

  // Report paragraph
  document.getElementById("modal-report").textContent =
    s.report_paragraph || "Not yet generated.";

  // Per-class probs
  const probs_div = document.getElementById("modal-class-probs");
  const sl = s.ai_result?.study_level || {};
  probs_div.innerHTML = Object.entries(sl).map(([col, vals]) => {
    const display = CLASS_DISPLAY[col] || col;
    const pos = vals.positive;
    return `<div class="class-prob-item ${pos ? "positive" : ""}">
      ${display}: <strong>${vals.prob.toFixed(3)}</strong>
      ${pos ? " &#9651;" : ""}
    </div>`;
  }).join("");

  // Metrics
  const m = s.metrics || {};
  document.getElementById("modal-metrics").innerHTML = [
    ["Prevalence",   fmt_pct(s.prevalence || 0), ""],
    ["PPV",          fmt_pct(m.ppv || 0), m.ppv > 0.5 ? "good" : ""],
    ["NPV",          fmt_pct(m.npv || 0), "good"],
    ["Pain Index",   (m.pain_index || 0).toFixed(2), "pain"],
    ["Sensitivity",  "98.0%", ""],
    ["Specificity",  "97.0%", ""],
  ].map(([label, value, cls]) =>
    `<div class="metric-card">
      <div class="label">${label}</div>
      <div class="value ${cls}">${value}</div>
    </div>`
  ).join("");

  // Confusion matrix
  const tp = m.tp ?? "—", fp = m.fp ?? "—";
  const fn = m.fn ?? "—", tn = m.tn ?? "—";
  document.getElementById("modal-confusion").innerHTML = `
    <tr>
      <th></th><th>AI Positive</th><th>AI Negative</th>
    </tr>
    <tr>
      <th>True ICH</th>
      <td class="tp">TP = ${tp}</td>
      <td class="fn">FN = ${fn}</td>
    </tr>
    <tr>
      <th>No ICH</th>
      <td class="fp">FP = ${fp}</td>
      <td class="tn">TN = ${tn}</td>
    </tr>`;

  document.getElementById("modal").classList.add("open");
}

function closeModal() {
  document.getElementById("modal").classList.remove("open");
}

document.getElementById("modal").addEventListener("click", function(e) {
  if (e.target === this) closeModal();
});

// ── Viewer ────────────────────────────────────────────────────────────────────

let _vStudy    = null;
let _vHotIdx   = 0;
let _vHotTotal = 0;

const CLASS_DISPLAY_LONG = {
  epidural:         "Epidural hematoma",
  intraparenchymal: "Intraparenchymal hemorrhage",
  intraventricular: "Intraventricular hemorrhage",
  subarachnoid:     "Subarachnoid hemorrhage",
  subdural:         "Subdural hematoma",
  any:              "Any intracranial hemorrhage",
};

function openViewer(json_str) {
  const s = JSON.parse(json_str);
  _vStudy  = s;
  _vHotIdx = 0;
  const hot = (s.ai_result?.hot_slices || []);
  _vHotTotal = hot.length;

  document.getElementById("viewer-title").textContent =
    (s.patient_name || s.patient_id) + "  —  " + (s.description || "CT HEAD");

  // Impression
  const imp = s.impression_bullet || (s.ai_positive ? "AI POSITIVE" : "AI negative");
  const impEl = document.getElementById("sr-impression");
  impEl.textContent = imp;
  impEl.className = "sr-impression" + (s.ai_positive ? " positive" : "");

  // Per-class probability bars
  const sl = s.ai_result?.study_level || {};
  const ORDER = ["any","subdural","epidural","intraparenchymal","intraventricular","subarachnoid"];
  document.getElementById("sr-probs").innerHTML = ORDER.map(col => {
    const v   = sl[col];
    if (!v) return "";
    const pct = (v.prob * 100).toFixed(1);
    const pos = v.positive;
    return `<div class="prob-bar-row">
      <div class="prob-bar-label">${CLASS_DISPLAY_LONG[col] || col}</div>
      <div class="prob-bar-track">
        <div class="prob-bar-fill ${pos ? "positive" : ""}" style="width:${pct}%"></div>
      </div>
      <div class="prob-bar-val">${pct}%</div>
    </div>`;
  }).join("");

  // Metrics
  const m  = s.metrics || {};
  const ms = m.metrics_source || "";
  const lr_pos = m.lr_positive != null ? m.lr_positive.toFixed(1) : (m.sensitivity && m.specificity ? (m.sensitivity / (1 - m.specificity + 0.0001)).toFixed(1) : "—");
  const lr_neg = m.lr_negative != null ? m.lr_negative.toFixed(3) : "—";
  document.getElementById("sr-metrics").innerHTML = [
    ["Prevalence", fmt_pct(s.prevalence || 0), ""],
    ["PPV",        fmt_pct(m.ppv || 0),        ""],
    ["NPV",        fmt_pct(m.npv || 0),        ""],
    ["Sensitivity",fmt_pct(m.sensitivity || 0),""],
    ["Specificity",fmt_pct(m.specificity || 0),""],
    ["LR+",        lr_pos,                      ""],
    ["LR−",        lr_neg,                      ""],
    ["Pain Index", (m.pain_index || 0).toFixed(2), "pain",
     "FP:TP ratio — incorrect AI positive calls the radiologist must review and override per true ICH detected"],
    ["AUC",        m.mean_auc ? m.mean_auc.toFixed(4) : "—", "", ""],
  ].map(([label, value, cls, tip]) =>
    `<div class="sr-metric" ${tip ? `title="${tip}"` : ""}>
      <div class="label">${label}</div>
      <div class="value ${cls}">${value}</div>
    </div>`
  ).join("");
  document.getElementById("sr-metrics-source").textContent = ms ? "Source: " + ms : "";

  // Report
  document.getElementById("sr-report").textContent =
    s.report_paragraph || "Report not yet generated.";

  // Disclaimer
  document.getElementById("sr-disclaimer").textContent =
    "AI-assisted screening tool. Not FDA-cleared. All findings require radiologist confirmation.";

  // Reset explainer and HU overlay for each new study
  document.getElementById("bayes-explainer").style.display = "none";
  _huOverlay = false;
  document.getElementById("hu-overlay-btn").textContent     = "HU Overlay";
  document.getElementById("hu-overlay-btn").style.background = "#b71c1c";
  document.getElementById("hu-overlay-btn").style.boxShadow  = "none";
  document.getElementById("hu-overlay-caption").style.display = "none";

  _loadSlice();
  document.getElementById("viewer").classList.add("open");
}

function _loadSlice() {
  const hot = (_vStudy?.ai_result?.hot_slices || []);
  const h   = hot[_vHotIdx] || {};

  // Image
  const uid  = _vStudy?.study_uid || "";
  const src  = `/api/slice_image?study_uid=${encodeURIComponent(uid)}&hot_idx=${_vHotIdx}`
             + `&overlay=${_huOverlay ? 1 : 0}&_=${Date.now()}`;
  document.getElementById("viewer-img").src = src;

  // Slice info
  const cls = h.dominant_class || _vStudy?.dominant_class || "";
  const z   = h.slice_z_mm  != null ? `z = ${h.slice_z_mm > 0 ? "+" : ""}${h.slice_z_mm.toFixed(1)} mm` : "";
  const pr  = h.prob        != null ? `model score ${(h.prob * 100).toFixed(1)}%` : "";
  const idx_label = _vHotTotal > 0
    ? `Positive slice ${_vHotIdx + 1} of ${_vHotTotal}`
    : "No hot slices";
  document.getElementById("viewer-slice-info").innerHTML =
    `<strong>${idx_label}</strong><br>${cls}  ${pr}<br>${z}`;

  // Demo note if no real DICOM
  const hasDicom = !!(_vStudy?.ai_result?.series_folder);
  document.getElementById("viewer-demo-note").textContent =
    hasDicom ? "" : "Demo image — not from actual DICOM data";

  // Nav buttons
  document.getElementById("nav-prev").disabled = (_vHotIdx <= 0);
  document.getElementById("nav-next").disabled = (_vHotIdx >= _vHotTotal - 1);
}

function navigateSlice(delta) {
  _vHotIdx = Math.max(0, Math.min(_vHotIdx + delta, _vHotTotal - 1));
  _loadSlice();
}

function closeViewer() {
  document.getElementById("viewer").classList.remove("open");
}

let _huOverlay = false;

function toggleHuOverlay() {
  _huOverlay = !_huOverlay;
  const btn     = document.getElementById("hu-overlay-btn");
  const caption = document.getElementById("hu-overlay-caption");
  btn.textContent       = _huOverlay ? "HU Overlay ON" : "HU Overlay";
  btn.style.background  = _huOverlay ? "#4a0000" : "#b71c1c";
  btn.style.boxShadow   = _huOverlay ? "0 0 0 2px #ef9a9a" : "none";
  caption.style.display = _huOverlay ? "block" : "none";
  _loadSlice();
}

function toggleBayesExplainer() {
  const el  = document.getElementById("bayes-explainer");
  const btn = event.currentTarget;
  const visible = el.style.display !== "none";
  el.style.display = visible ? "none" : "block";
  btn.textContent  = visible ? "In-Depth Explanation" : "Hide Explanation";
  btn.style.background = visible ? "#546e7a" : "#37474f";
}

function copyReport() {
  const text = document.getElementById("sr-report").textContent || "";
  if (!text || text === "Report not yet generated.") return;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.getElementById("copy-report-btn");
    const orig = btn.textContent;
    btn.textContent = "Copied ✓";
    btn.style.background = "#2e7d32";
    setTimeout(() => {
      btn.textContent = orig;
      btn.style.background = "#1565c0";
    }, 2000);
  });
}

document.getElementById("viewer").addEventListener("click", function(e) {
  if (e.target === this) closeViewer();
});

document.addEventListener("keydown", function(e) {
  if (!document.getElementById("viewer").classList.contains("open")) return;
  if (e.key === "ArrowLeft")  navigateSlice(-1);
  if (e.key === "ArrowRight") navigateSlice(+1);
  if (e.key === "Escape")     closeViewer();
});

// Initial load + auto-refresh
loadWorklist();
setInterval(loadWorklist, 15000);
</script>
</body>
</html>
"""


# ── Slice image helpers ───────────────────────────────────────────────────────

def _png_bytes(img) -> bytes:
    """PIL Image → PNG bytes."""
    import io as _io
    buf = _io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _no_image_png(msg: str = "No image") -> bytes:
    """Return a dark 512×512 placeholder PNG with centred text."""
    img = Image.new("RGB", (512, 512), color=(20, 20, 30))
    draw = ImageDraw.Draw(img)
    w, h = img.size
    draw.text((w // 2, h // 2), msg, fill=(100, 100, 120), anchor="mm")
    return _png_bytes(img)


def _synthetic_brain_png(hot_slice: dict, dominant_class: str) -> bytes:
    """
    Generate a synthetic grayscale brain CT slice for demo use.

    The image is NOT diagnostic — it is a geometric approximation that
    conveys the spatial relationship of normal brain tissue and the AI-detected
    abnormality class, for demonstration purposes only.
    """
    import math

    SZ      = 512
    CX, CY  = SZ // 2, SZ // 2
    R_skull = 220   # outer skull radius
    R_brain = 200   # brain surface radius
    cls     = dominant_class.lower()

    # z_mm rotates the finding around the brain between slices
    z   = hot_slice.get("slice_z_mm", 0) or 0
    ang = (z * 5.0) % 360   # degrees

    def arc_polygon(r_out, r_in, a_start, a_end, steps=80):
        """Points for a filled annular sector (crescent / wedge shape)."""
        pts = []
        for i in range(steps + 1):
            a = math.radians(a_start + (a_end - a_start) * i / steps)
            pts.append((CX + r_out * math.cos(a), CY + r_out * math.sin(a)))
        for i in range(steps, -1, -1):
            a = math.radians(a_start + (a_end - a_start) * i / steps)
            pts.append((CX + r_in  * math.cos(a), CY + r_in  * math.sin(a)))
        return pts

    img  = Image.new("L", (SZ, SZ), 0)
    draw = ImageDraw.Draw(img)

    # ── Base anatomy ──────────────────────────────────────────────────────────

    # Skull (bright bone)
    draw.ellipse([CX - R_skull, CY - R_skull, CX + R_skull, CY + R_skull],
                 fill=230)

    # Subdural / subarachnoid space (slightly darker than skull)
    draw.ellipse([CX - R_brain - 10, CY - R_brain - 10,
                  CX + R_brain + 10, CY + R_brain + 10], fill=180)

    # Brain parenchyma (medium gray)
    draw.ellipse([CX - R_brain, CY - R_brain, CX + R_brain, CY + R_brain],
                 fill=105)

    # Gray-white differentiation: slightly brighter cortical ribbon
    R_cortex = R_brain - 18
    draw.ellipse([CX - R_cortex, CY - R_cortex, CX + R_cortex, CY + R_cortex],
                 fill=92)

    # Ventricles (CSF, dark)
    draw.ellipse([CX - 30, CY - 40, CX + 30, CY - 5],   fill=35)  # lateral L/R
    draw.ellipse([CX - 8,  CY - 15, CX + 8,  CY + 5],   fill=35)  # 3rd

    # Falx (midline dense structure, faint)
    draw.line([(CX, CY - R_brain), (CX, CY + 10)], fill=140, width=2)

    # ── Abnormality overlay ───────────────────────────────────────────────────

    if "subdural" in cls:
        # Large hyperdense crescent — exaggerated for demo visibility.
        # Fills from inner skull surface inward by ~35px, spanning 130°.
        # Represents a moderate–large subdural with mild mass effect.
        a0 = ang - 65
        a1 = ang + 65
        pts = arc_polygon(R_skull - 4, R_brain - 28, a0, a1)
        draw.polygon(pts, fill=245)
        # Re-draw brain cortex over the inner edge so it looks like
        # the brain is being pushed away from the hematoma
        pts2 = arc_polygon(R_brain - 28, R_brain - 40, a0, a1)
        draw.polygon(pts2, fill=95)

    elif "epidural" in cls:
        # Biconvex (lenticular) haematoma — bounded by skull and dura.
        # Thicker in the centre, tapering at the ends — classic epidural shape.
        a0 = ang - 35
        a1 = ang + 35
        # Outer arc at skull, inner arc bows inward more at the midpoint
        pts = []
        import math as _m
        steps = 80
        for i in range(steps + 1):
            a = _m.radians(a0 + (a1 - a0) * i / steps)
            pts.append((CX + (R_skull - 4) * _m.cos(a),
                        CY + (R_skull - 4) * _m.sin(a)))
        # Inner edge: parabolic bulge — closer to brain at midpoint
        for i in range(steps, -1, -1):
            a    = _m.radians(a0 + (a1 - a0) * i / steps)
            frac = 1 - (2 * i / steps - 1) ** 2   # 0→1→0
            r_in = R_brain + 8 + frac * 30          # max 38px inward
            pts.append((CX + r_in * _m.cos(a), CY + r_in * _m.sin(a)))
        draw.polygon(pts, fill=248)

    elif "intraparenchymal" in cls:
        # Round hyperdense haematoma within parenchyma
        rad = math.radians(ang)
        ox  = int(CX + 65 * math.cos(rad))
        oy  = int(CY + 65 * math.sin(rad))
        draw.ellipse([ox - 26, oy - 26, ox + 26, oy + 26], fill=248)
        # Oedema halo (slightly brighter than parenchyma)
        draw.ellipse([ox - 38, oy - 38, ox + 38, oy + 38], fill=125)
        draw.ellipse([ox - 26, oy - 26, ox + 26, oy + 26], fill=248)

    elif "intraventricular" in cls:
        # Blood layering in ventricles (hyperdense fluid level)
        draw.ellipse([CX - 30, CY - 40, CX + 30, CY - 5],  fill=200)
        draw.ellipse([CX - 20, CY - 40, CX + 20, CY - 25], fill=235)

    elif "subarachnoid" in cls:
        # SAH: blood tracking in cisterns and sulci — bright ring just outside brain
        for t in range(0, 360, 6):
            a  = math.radians(t)
            jitter = math.sin(t * 0.3) * 4
            rx = int(CX + (R_brain + 5 + jitter) * math.cos(a))
            ry = int(CY + (R_brain + 5 + jitter) * math.sin(a))
            draw.ellipse([rx - 5, ry - 5, rx + 5, ry + 5], fill=220)
        # Sylvian fissure fill
        draw.ellipse([CX - 40, CY - 5, CX - 10, CY + 15], fill=215)
        draw.ellipse([CX + 10, CY - 5, CX + 40, CY + 15], fill=215)

    # Subtle Gaussian blur to soften hard edges
    img = img.filter(ImageFilter.GaussianBlur(radius=1.0))

    # ── Annotations ───────────────────────────────────────────────────────────
    rgb   = Image.merge("RGB", [img, img, img])
    draw2 = ImageDraw.Draw(rgb)

    # Bottom bar
    prob  = hot_slice.get("prob", 0)
    z_mm  = hot_slice.get("slice_z_mm", 0)
    label = f"z = {z_mm:+.1f} mm     {dominant_class}     score {prob*100:.1f}%"
    draw2.rectangle([0, SZ - 24, SZ, SZ], fill=(0, 0, 0))
    draw2.text((SZ // 2, SZ - 12), label, fill=(255, 220, 50), anchor="mm")

    # DEMO watermark
    draw2.text((8, 8), "DEMO — NOT DIAGNOSTIC", fill=(200, 50, 50))

    return _png_bytes(rgb)


def _dicom_slice_png(series_folder: str, sop_uid: str, slice_idx: int,
                     hu_overlay: bool = False) -> bytes | None:
    """
    Read a real DICOM slice, apply medium HU window, return PNG bytes.
    If hu_overlay=True, pixels in the acute ICH HU range (48–90 HU) are
    tinted red as a physics-based density cue independent of the AI model.
    Returns None if the file cannot be read.
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pydicom, warnings
        from hu_windows import apply_window, WINDOW_MEDIUM

        folder = Path(series_folder)
        dcm_files = sorted(folder.glob("*.dcm"))
        if not dcm_files:
            return None

        target = None
        if sop_uid:
            for f in dcm_files:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
                    if str(getattr(ds, "SOPInstanceUID", "")) == sop_uid:
                        target = f
                        break
                except Exception:
                    continue
        if target is None and 0 <= slice_idx < len(dcm_files):
            target = dcm_files[slice_idx]
        if target is None:
            return None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = pydicom.dcmread(str(target))

        arr = ds.pixel_array.astype(np.float32)
        slope     = float(getattr(ds, "RescaleSlope",     1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        hu  = arr * slope + intercept
        win = apply_window(hu, WINDOW_MEDIUM)  # float32 [0,1]
        px  = (win * 255).astype(np.uint8)

        img = Image.fromarray(px, mode="L").resize((512, 512), Image.LANCZOS)
        rgb = Image.merge("RGB", [img, img, img])

        if hu_overlay:
            # Resample HU array to 512×512 to match display image
            from PIL import Image as _Im
            hu_img = _Im.fromarray(hu).resize((512, 512), _Im.LANCZOS)
            hu_512 = np.array(hu_img)

            # Mask: pixels in acute ICH density range (48–90 HU)
            ICH_HU_LOW, ICH_HU_HIGH = 48.0, 90.0
            mask = ((hu_512 >= ICH_HU_LOW) & (hu_512 <= ICH_HU_HIGH))

            # Blend red channel into masked pixels (semi-transparent tint)
            r, g, b = rgb.split()
            r_arr = np.array(r)
            g_arr = np.array(g)
            b_arr = np.array(b)
            # Boost red, suppress green/blue on mask pixels
            r_arr[mask] = np.clip(r_arr[mask].astype(np.int16) + 120, 0, 255).astype(np.uint8)
            g_arr[mask] = (g_arr[mask] * 0.35).astype(np.uint8)
            b_arr[mask] = (b_arr[mask] * 0.35).astype(np.uint8)
            rgb = Image.merge("RGB", [
                Image.fromarray(r_arr),
                Image.fromarray(g_arr),
                Image.fromarray(b_arr),
            ])

        return _png_bytes(rgb)

    except Exception:
        return None


def _slice_png(study: dict, hot_idx: int, hu_overlay: bool = False) -> bytes:
    """Return PNG bytes for the hot slice at hot_idx, real or synthetic."""
    if not _PIL_OK:
        return b""

    ar         = study.get("ai_result", {})
    hot_slices = ar.get("hot_slices", [])
    dominant   = study.get("dominant_class", "") or ar.get("dominant_class", "")

    if not hot_slices:
        return _no_image_png("No hot slices recorded")

    hot_idx = max(0, min(hot_idx, len(hot_slices) - 1))
    hot     = hot_slices[hot_idx]

    series_folder = ar.get("series_folder", "")
    if series_folder and Path(series_folder).is_dir():
        sop_uid   = hot.get("sop_uid", "")
        slice_idx = hot.get("slice_index", 0)
        png = _dicom_slice_png(series_folder, sop_uid, slice_idx, hu_overlay)
        if png:
            return png

    # Synthetic demo fallback (overlay not applied to synthetic images)
    return _synthetic_brain_png(hot, dominant)


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/studies")
def api_studies():
    return jsonify(get_studies())


@app.route("/api/slice_image")
def api_slice_image():
    study_uid  = request.args.get("study_uid", "")
    hot_idx    = int(request.args.get("hot_idx", "0"))
    hu_overlay = request.args.get("overlay", "0") == "1"
    studies    = get_studies()
    study      = next((s for s in studies if s.get("study_uid") == study_uid), None)
    if not study:
        return Response(_no_image_png("Study not found"), mimetype="image/png")
    return Response(_slice_png(study, hot_idx, hu_overlay), mimetype="image/png")


@app.route("/api/studies", methods=["POST"])
def api_add_study():
    study = request.get_json()
    result = add_study(study)
    return jsonify(result), 201


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ICH AI Worklist server")
    parser.add_argument("--port",     type=int, default=5050)
    parser.add_argument("--host",     default="0.0.0.0")
    parser.add_argument("--add-demo", action="store_true",
                        help="Seed worklist with demo studies")
    args = parser.parse_args()

    if args.add_demo:
        print("Seeding worklist with demo studies...")
        for s in DEMO_STUDIES:
            add_study(s)
        print(f"Added {len(DEMO_STUDIES)} demo studies → {WORKLIST_PATH}")

    print(f"\nICH AI Worklist running at http://{args.host}:{args.port}")
    print(f"Worklist file: {WORKLIST_PATH}\n")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
