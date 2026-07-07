#!/usr/bin/env python3
"""
compare_studies.py — the comparison tool the agent calls on the follow-up path.

Given two study directories (prior, current), re-loads BOTH volumes from disk
(server-side; pixel data is never passed through the language model) and returns
the structured interval deltas the comparative report is built from:

    midline shift (mm), SDH cross-sectional area (mm^2, stability proxy),
    ipsilateral white-matter attenuation (HU, edema proxy)

Honesty boundary: on this phantom the "measurements" are exact by construction.
On real studies, registered quantitative comparison (co-registration +
segmentation of both exams) is a separate pipeline; this tool is where that
pipeline would be called. The agent ORCHESTRATES and NARRATES the delta; it
does not itself measure it.
"""
import glob
import json
import os
import sys

import numpy as np
import pydicom

sys.path.insert(0, os.path.dirname(__file__))
import synthetic_ct as s


def _load_volume(series_dir):
    files = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    vol = []
    for f in files:
        d = pydicom.dcmread(f)
        vol.append(d.pixel_array.astype(float) * float(d.RescaleSlope)
                   + float(d.RescaleIntercept))
    return np.stack(vol)


def compare(prior_dir, current_dir, sdh_side="left"):
    pv, cv = _load_volume(prior_dir), _load_volume(current_dir)
    ms_p, ms_c = (s.measure_midline_shift_mm(pv), s.measure_midline_shift_mm(cv))
    ar_p, ar_c = (s.measure_sdh_area_mm2(pv, sdh_side),
                  s.measure_sdh_area_mm2(cv, sdh_side))
    wm_p, wm_c = (s.measure_wm_hu(pv, sdh_side), s.measure_wm_hu(cv, sdh_side))

    def band(delta, tol, up, down, flat):
        return up if delta > tol else down if delta < -tol else flat

    return {
        "sdh_side": sdh_side,
        "midline_shift_mm": {"prior": round(abs(ms_p), 1),
                             "current": round(abs(ms_c), 1),
                             "delta": round(abs(ms_c) - abs(ms_p), 1),
                             "interpretation": band(
                                 abs(ms_c) - abs(ms_p), 1.0,
                                 "new/increased mass effect", "improved",
                                 "unchanged")},
        "sdh_area_mm2": {"prior": round(ar_p), "current": round(ar_c),
                         "delta": round(ar_c - ar_p),
                         "interpretation": band(
                             ar_c - ar_p, 0.15 * max(ar_p, 1),
                             "hematoma enlarging", "hematoma decreasing",
                             "hematoma stable in size")},
        "ipsilateral_wm_hu": {"prior": round(wm_p), "current": round(wm_c),
                              "delta": round(wm_c - wm_p),
                              "interpretation": band(
                                  wm_c - wm_p, 3.0,
                                  "increased attenuation",
                                  "decreased attenuation — developing edema",
                                  "unchanged")},
    }


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    idx = json.load(open(os.path.join(here, "pacs_index.json")))
    prior = next(s for s in idx if s["accession"] == "DEMOACC001")
    curr = next(s for s in idx if s["accession"] == "DEMOACC002")
    result = compare(prior["series_dir"], curr["series_dir"])
    print(json.dumps(result, indent=2))
