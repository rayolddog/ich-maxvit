#!/usr/bin/env python3
"""
bloodness_speed.py — hemorrhage-annotation preprocessing for ITK-SNAP & 3D Slicer.

A PERSONAL productivity tool: turn one NCCT head study into domain-specific
inputs that make the approved annotation tools fast for hemorrhage. Introduces
no new software — it emits standard NIfTI that ITK-SNAP and Slicer already
ingest. The annotator seeds and corrects; this just gives the tools a
blood-aware field to work in instead of generic intensity.

Two outputs (both geometry-preserving NIfTI, overlay-aligned to the source):

  <out>_speed.nii.gz   continuous blood-probability in [0,1]
                       -> ITK-SNAP: Active Contour, "pre-segmentation /
                          probability" feature image. The snake flows into
                          blood and stops at the inner table.
  <out>_seed.nii.gz    discrete candidate label (1 = blood candidate)
                       + bone as label 2
                       -> 3D Slicer: import as segmentation, then correct with
                          Segment Editor (Grow-from-seeds / Fill-between-slices).

Method (all in calibrated HU; nothing is windowed/saturated):
  1. blood membership: smooth trapezoid, 1.0 in [50,70] HU, 0 outside [40,80]
  2. partial-volume-rim removal by THICKNESS, not distance. Real subdural blood
     TOUCHES the inner table exactly like the rim does, so a bone-distance gate
     would eat the blood at the interface where it actually lives. The true
     discriminator is radial extent: the PV rim is ~1 voxel deep, real blood is
     a multi-voxel band. A per-slice morphological opening (disk ~1.5 mm) erases
     structures thinner than the disk in any direction — deleting the thin rim
     while keeping the thick collection, even where it hugs bone.
  3. exclude bone (HU>300) and air/low (HU<0).

Usage (run with the repo venv that has SimpleITK):
  .venv/bin/python annotation_assist/bloodness_speed.py <dicom_series_dir> \
        --out annotation_assist/out/<name>
  ... --selftest        # synthetic check, no data needed
"""
import argparse
import os

import numpy as np


def blood_membership(hu):
    """Smooth trapezoid for ACUTE (hyperdense) blood. Plateau 52-72 HU; ramps
    46->52 and 72->80. The lower edge sits deliberately ABOVE gray matter
    (~37 HU, up to ~45 with noise/partial-volume) so normal cortex is excluded
    — the over-inclusion that a 40-HU floor produces. NOTE: this targets ACUTE
    blood; subacute/chronic collections become iso-/hypodense and are out of
    scope for this membership (they need a different, lower band)."""
    m = np.zeros_like(hu, dtype=np.float32)
    up = (hu >= 46) & (hu < 52)
    m[up] = (hu[up] - 46) / 6.0
    m[(hu >= 52) & (hu <= 72)] = 1.0
    dn = (hu > 72) & (hu <= 80)
    m[dn] = (80 - hu[dn]) / 8.0
    return m


def _disk(radius_px):
    r = int(max(1, round(radius_px)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def remove_thin_rim(binary, radius_px):
    """Per-slice 2D morphological opening: erase structures thinner than the
    disk (the ~1-voxel inner-table rim) while keeping thick collections. 2D per
    axial slice handles anisotropic (thick-slice) data correctly — the rim is a
    per-slice shell."""
    from scipy import ndimage
    disk = _disk(radius_px)
    out = np.zeros_like(binary)
    for z in range(binary.shape[0]):
        out[z] = ndimage.binary_opening(binary[z], structure=disk)
    return out


def intracranial_mask(hu, spacing_mm):
    """Fill the intracranial cavity so extracranial muscle (also ~40-60 HU) is
    excluded. Per-slice: close the skull ring, fill its interior, keep brain-
    range voxels; then the largest 3D component. 2D fill works above the skull
    base (where SDH is interpretable) and degrades at the base (where SDH is
    near-uninterpretable anyway — a known radiologist blind spot), which is an
    acceptable trade."""
    from scipy import ndimage
    sz, sy, sx = spacing_mm
    bone = hu > 300
    disk = _disk(max(1, round(2.0 / sx)))
    cav = np.zeros_like(bone)
    for z in range(bone.shape[0]):
        closed = ndimage.binary_closing(bone[z], structure=disk)
        cav[z] = ndimage.binary_fill_holes(closed) & ~closed
    intra = cav & (hu > -50) & (hu < 100)
    lab, n = ndimage.label(intra)
    if n:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        intra = lab == int(sizes.argmax())
    return intra


def bloodness(hu, spacing_mm, open_mm=1.5):
    """spacing_mm = (z, y, x). Returns (speed[0..1], bone_mask, intracranial)."""
    sz, sy, sx = spacing_mm
    m = blood_membership(hu)
    bone = hu > 300
    intra = intracranial_mask(hu, spacing_mm)
    binary = (m >= 0.5) & ~bone & intra
    kept = remove_thin_rim(binary, radius_px=open_mm / sx)
    speed = (m * kept).astype(np.float32)
    speed[~intra] = 0.0
    return speed, bone, intra


def selftest():
    # The hard, honest test: BOTH the real SDH and the PV rim touch bone.
    # Only thickness distinguishes them (6 mm band vs 1-voxel shell). A distance
    # gate would fail this; the opening should pass it.
    sp = (5.0, 0.5, 0.5)                               # (z,y,x) mm
    hu = np.full((4, 220, 220), 30.0, np.float32)     # brain
    yy, xx = np.mgrid[0:220, 0:220]
    r = np.hypot(yy - 110, xx - 110)
    hu[:, r > 100] = 1100                              # bone (inner table r=100)
    # region A: real SDH, 6 mm thick (12 px), hugging bone at r 88-100
    hu[:, (r > 88) & (r <= 100)] = 62
    # region B (separate quadrant): a 1-voxel PV rim just inside bone
    rim = (r > 99) & (r <= 100) & (xx > 110)          # right side only
    hu[:, rim] = 60
    # carve region A to the LEFT so the two don't merge
    hu[:, ((r > 88) & (r <= 100)) & (xx > 110)] = 30
    speed, bone, intra = bloodness(hu, sp)
    real = speed[:, (r > 90) & (r < 98) & (xx < 110)].mean()
    rimv = speed[:, rim].mean()
    print(f"  real-SDH mean speed {real:.2f} | 1-voxel PV-rim mean speed {rimv:.3f}")
    assert real > 0.6, "real SDH (thick, touching bone) should be retained"
    assert rimv < 0.15, "1-voxel PV rim should be removed by the opening"
    print("  SELFTEST PASS: thick blood kept even against bone; thin rim removed")


def run(series_dir, out_base):
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(series_dir)
    if not files:
        raise SystemExit(f"no DICOM series found in {series_dir}")
    reader.SetFileNames(files)
    img = reader.Execute()                # GDCM decodes JPEG-Lossless, keeps geom
    hu = sitk.GetArrayFromImage(img).astype(np.float32)   # (z,y,x), already HU
    sx, sy, sz = img.GetSpacing()          # (x,y,z) mm

    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)

    def _write(arr, path, is_label):
        o = sitk.GetImageFromArray(arr)
        o.CopyInformation(img)             # exact overlay geometry
        sitk.WriteImage(sitk.Cast(o, sitk.sitkUInt8 if is_label
                                  else sitk.sitkFloat32), path)

    speed, bone, intra = bloodness(hu, (sz, sy, sx))
    _write(speed, out_base + "_speed.nii.gz", False)
    cand = speed >= 0.5
    seed = np.zeros_like(hu, np.uint8)
    seed[bone] = 2
    seed[cand] = 1
    _write(seed, out_base + "_seed.nii.gz", True)

    vox_ml = (sx * sy * sz) / 1000.0
    vol_ml = float(cand.sum()) * vox_ml
    print(f"[bloodness] {os.path.basename(series_dir)}: "
          f"{hu.shape[0]} slices, spacing {sx:.2f}x{sy:.2f}x{sz:.2f} mm")
    print(f"[bloodness]   total candidate volume {vol_ml:.1f} mL")

    # connected components — the annotator seeds ONE lesion, so the largest
    # connected component is the likely target; scattered small ones are
    # contamination (choroid, vessels, gray-matter speckle).
    from scipy import ndimage
    lab, n = ndimage.label(cand)
    if n:
        sizes = np.bincount(lab.ravel())[1:]
        order = np.argsort(sizes)[::-1]
        print(f"[bloodness]   {n} connected components; largest few (mL):")
        for k in order[:5]:
            print(f"      component {k+1}: {sizes[k]*vox_ml:.1f} mL")
        largest = order[0] + 1
        _qc_overlay(hu, cand, lab == largest, out_base + "_QC.png",
                    sizes[largest - 1] * vox_ml, (sz, sy, sx))
    print(f"    {out_base}_speed.nii.gz   -> ITK-SNAP active-contour feature")
    print(f"    {out_base}_seed.nii.gz    -> 3D Slicer segmentation seed")
    print(f"    {out_base}_QC.png         -> quick visual check")


def _qc_overlay(hu, cand, largest, path, largest_ml, spacing_mm):
    """MULTIPLANAR check through the largest component's centroid — axial,
    coronal, sagittal — because subfrontal and subtemporal subdurals lie
    parallel to the axial plane and are only interpretable on reformats
    (the reason radiologists check coronal/sagittal for these). red = all
    candidates on that plane; yellow = the largest component."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sz, sy, sx = spacing_mm
    zc, yc, xc = (int(round(c)) for c in ndimage_center(largest))
    # aspect = physical height / width of a pixel in each plane. Reformats have
    # z-rows (5 mm) that must display ~10x taller than in-plane (0.49 mm) columns.
    planes = [
        ("axial",    hu[zc], cand[zc], largest[zc], sy / sx),
        ("coronal",  hu[:, yc], cand[:, yc], largest[:, yc], sz / sx),
        ("sagittal", hu[:, :, xc], cand[:, :, xc], largest[:, :, xc], sz / sy),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    for a, (name, im, cd, lg, aspect) in zip(ax, planes):
        a.imshow(im, cmap="gray", vmin=-20, vmax=100, aspect=aspect)  # subdural W
        red = np.zeros((*im.shape, 4)); red[cd] = [1, 0, 0, 0.45]
        a.imshow(red, aspect=aspect)
        if lg.any():
            a.contour(lg, levels=[0.5], colors="yellow", linewidths=0.8)
        a.set_title(name); a.axis("off")
    fig.suptitle(f"largest intracranial component = {largest_ml:.1f} mL  "
                 f"(yellow); red = all candidates on the plane")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def ndimage_center(mask):
    from scipy import ndimage
    return ndimage.center_of_mass(mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("series_dir", nargs="?")
    ap.add_argument("--out", default="annotation_assist/out/study")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.series_dir:
        ap.error("provide a DICOM series directory (or --selftest)")
    run(a.series_dir, a.out)


if __name__ == "__main__":
    main()
