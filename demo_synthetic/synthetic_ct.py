#!/usr/bin/env python3
"""
synthetic_ct.py — minimal parametric head-CT phantom for the ICH demo.

Produces a 4-slice non-contrast head CT as a DICOM series. Real studies carry
200+ images, but pathology is usually shown on a handful; 4 slices keep the
demo cheap in storage and tokens while still carrying the pathology and the
short-interval deltas. The two CENTER slices carry the subdural hematoma.

THIS IS A DEMO FIXTURE, NOT A RADIOLOGICALLY REALISTIC IMAGE. It is a
labelled phantom that encodes *measurable deltas* (midline shift, white-matter
attenuation) for the comparison logic. No real patient data is used or derived.

Calibrated HU (approx adult NCCT):
    air -1000   CSF ~8   gray ~37   white ~28   white+edema ~22
    skull ~1100   acute blood (SDH) ~62

Demo pair (build_demo_case):
  BASELINE   : small left convexity SDH (5 mm thick, ~4 cm arc), no shift,
               normal white matter
  FOLLOW-UP  : SDH unchanged (5 mm), 4 mm midline shift, ipsilateral (left)
               white-matter hypoattenuation (28 -> 22 HU)
The classic short-interval subdural picture: mass effect and edema progress
while the collection itself looks unchanged.

Usage:
  synthetic_ct.py --out demo_synthetic/pacs     # writes both studies
  synthetic_ct.py --selftest                     # verify deltas, no write
"""
import argparse
import os

import numpy as np

FOV_MM = 230.0
N = 512
PX = FOV_MM / N
N_SLICES = 4
SLICE_MM = 5.0
SKULL_A, SKULL_B = 92.0, 108.0
SKULL_THK = 6.0
SDH_ARC_MM = 40.0          # ~4 cm along the inner table
SDH_CENTER_DEG = 135.0     # superolateral left convexity

HU = dict(air=-1000, csf=8, gray=37, white=28, white_edema=22,
          skull=1100, blood=62)


def _grid_mm():
    ax = (np.arange(N) - N / 2) * PX
    return np.meshgrid(ax, ax)     # X=cols, Y=rows (mm)


def make_slice(z_frac, sdh_side=None, sdh_mm=0.0, shift_mm=0.0,
               wm_edema_side=None):
    X, Y = _grid_mm()
    img = np.full((N, N), HU["air"], np.float32)

    taper = 0.62 + 0.38 * np.sin(np.pi * z_frac)
    a, b = SKULL_A * taper, SKULL_B * taper
    outer = (X / a) ** 2 + (Y / b) ** 2 <= 1.0
    inner = (X / (a - SKULL_THK)) ** 2 + (Y / (b - SKULL_THK)) ** 2 <= 1.0
    img[outer] = HU["skull"]
    img[inner] = HU["gray"]

    a_wm, b_wm = a - SKULL_THK - 12, b - SKULL_THK - 12
    wm = (X / a_wm) ** 2 + (Y / b_wm) ** 2 <= 1.0
    img[wm] = HU["white"]
    if wm_edema_side is not None:
        side = (X < 0) if wm_edema_side == "left" else (X > 0)
        img[wm & side] = HU["white_edema"]

    # ventricles (displaced by midline shift; mass effect pushes AWAY from SDH)
    for sgn in (-1, 1):
        vx, vy = sgn * 14 + shift_mm, -6
        vent = ((X - vx) / 8) ** 2 + ((Y - vy) / 16) ** 2 <= 1.0
        img[vent & inner] = HU["csf"]

    # subdural crescent on the two CENTER slices only
    if sdh_side is not None and sdh_mm > 0 and 0.25 < z_frac < 0.75:
        a_in, b_in = a - SKULL_THK, b - SKULL_THK
        a_sd, b_sd = a_in - sdh_mm, b_in - sdh_mm
        band = ((X / a_in) ** 2 + (Y / b_in) ** 2 <= 1.0) & \
               ((X / a_sd) ** 2 + (Y / b_sd) ** 2 > 1.0)
        theta = np.arctan2(Y, X)
        center = np.deg2rad(SDH_CENTER_DEG if sdh_side == "left"
                            else 180 - SDH_CENTER_DEG)
        half = (SDH_ARC_MM / 2) / (0.5 * (a_in + b_in))
        img[band & (np.abs(theta - center) < half)] = HU["blood"]

    return img


def build_study(sdh_mm, shift_mm, wm_edema_side, sdh_side="left"):
    return np.stack([make_slice((i + 0.5) / N_SLICES, sdh_side, sdh_mm,
                                 shift_mm, wm_edema_side)
                     for i in range(N_SLICES)])


# --- measurement helpers (shared with the comparison tool + selftest) -------
def measure_midline_shift_mm(vol):
    X, _ = _grid_mm()
    best, best_area = 0.0, -1
    for sl in vol:
        csf = np.abs(sl - HU["csf"]) < 3
        if csf.sum() > max(best_area, 50):
            best_area, best = csf.sum(), float(np.mean(X[csf]))
    return best


def measure_sdh_area_mm2(vol, side="left"):
    X, _ = _grid_mm()
    keep = (X < 0) if side == "left" else (X > 0)
    n = int(sum(((np.abs(sl - HU["blood"]) < 4) & keep).sum() for sl in vol))
    return n * PX * PX


def measure_wm_hu(vol, side="left"):
    X, _ = _grid_mm()
    keep = (X < 0) if side == "left" else (X > 0)
    vals = [np.median(sl[(sl > 15) & (sl < 33) & keep])
            for sl in vol if ((sl > 15) & (sl < 33) & keep).sum() > 100]
    return float(np.median(vals)) if vals else float("nan")


def selftest():
    base = build_study(5.0, 0.0, None)
    fup = build_study(5.0, 4.0, "left")
    ms = (measure_midline_shift_mm(base), measure_midline_shift_mm(fup))
    ar = (measure_sdh_area_mm2(base), measure_sdh_area_mm2(fup))
    wm = (measure_wm_hu(base, "left"), measure_wm_hu(fup, "left"))
    print(f"  midline shift:  {ms[0]:+.1f} -> {ms[1]:+.1f} mm  (delta {ms[1]-ms[0]:+.1f})")
    print(f"  SDH area:       {ar[0]:.0f} -> {ar[1]:.0f} mm2  (delta {ar[1]-ar[0]:+.0f}, ~stable)")
    print(f"  left WM median: {wm[0]:.0f} -> {wm[1]:.0f} HU   (delta {wm[1]-wm[0]:+.0f}, decrease)")
    assert abs(ms[1] - ms[0]) > 2.5, "midline shift not encoded"
    assert abs(ar[1] - ar[0]) < 0.15 * max(ar), "SDH area should be ~stable"
    assert (wm[0] - wm[1]) > 3.0, "WM hypoattenuation not encoded"
    print("  SELFTEST PASS: three deltas measurable, correctly signed")


def build_demo_case(out_dir):
    from dicom_writer import write_study
    meta = {}
    meta["baseline"] = write_study(build_study(5.0, 0.0, None),
                                   out_dir, "baseline")
    meta["followup"] = write_study(build_study(5.0, 4.0, "left"),
                                   out_dir, "followup")
    print(f"[synthetic_ct] wrote baseline + follow-up -> {out_dir}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "pacs"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    build_demo_case(args.out)


if __name__ == "__main__":
    main()
