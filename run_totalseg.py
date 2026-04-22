"""
Run TotalSegmentator on a CT head study from the demo_studies folder.

Steps:
  1. Read DICOM slices from a study folder, sort by z position
  2. Stack into a 3D NIfTI volume
  3. Run TotalSegmentator (brain_structures task)
  4. Display segmentation overlaid on CT slices with matplotlib

Usage:
  python run_totalseg.py [--study STUDY_FOLDER] [--slice N]

Example:
  python run_totalseg.py --study demo_studies/positive/subarachnoid__ID_9fae411ae8
"""

import argparse
import os
import tempfile
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # must be set before importing pyplot

import numpy as np
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap


# ── DICOM → NIfTI ────────────────────────────────────────────────────────────

def load_dicom_series(study_dir: str) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Read all DICOM slices in a folder, sort by z, return (hu_volume, nifti)."""
    dcm_files = sorted(Path(study_dir).glob("*.dcm"))
    if not dcm_files:
        raise FileNotFoundError(f"No .dcm files found in {study_dir}")

    slices = []
    for f in dcm_files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = pydicom.dcmread(str(f), force=True)
        slices.append(ds)

    # Sort by ImagePositionPatient z, fall back to InstanceNumber
    def sort_key(ds):
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            return float(getattr(ds, "InstanceNumber", 0))

    slices.sort(key=sort_key)

    # Stack pixel arrays → HU
    arrays = []
    for ds in slices:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arrays.append(arr * slope + intercept)

    volume = np.stack(arrays, axis=2)          # (rows, cols, slices)

    ds0 = slices[0]

    # Pixel spacing: [row_spacing, col_spacing]
    try:
        ps = [float(x) for x in ds0.PixelSpacing]
    except Exception:
        ps = [1.0, 1.0]

    # ImageOrientationPatient: [row_cos_x, row_cos_y, row_cos_z,
    #                           col_cos_x, col_cos_y, col_cos_z]
    # row_cos = direction of increasing column index (across image)
    # col_cos = direction of increasing row index (down image)
    try:
        iop = [float(x) for x in ds0.ImageOrientationPatient]
        row_cos = np.array(iop[:3])   # step direction along columns (axis 1)
        col_cos = np.array(iop[3:])   # step direction along rows    (axis 0)
    except Exception:
        row_cos = np.array([1.0, 0.0, 0.0])
        col_cos = np.array([0.0, 1.0, 0.0])

    # Origin: patient coordinates of voxel (0, 0) in first slice
    try:
        ipp0 = np.array([float(x) for x in ds0.ImagePositionPatient])
    except Exception:
        ipp0 = np.zeros(3)

    # Slice direction: vector between consecutive slice origins
    if len(slices) > 1:
        try:
            ipp1 = np.array([float(x) for x in slices[1].ImagePositionPatient])
            slice_vec = ipp1 - ipp0
        except Exception:
            slice_vec = np.cross(row_cos, col_cos) * float(
                getattr(ds0, "SliceThickness", 5.0))
    else:
        slice_vec = np.cross(row_cos, col_cos) * float(
            getattr(ds0, "SliceThickness", 5.0))

    # Build affine: maps voxel (i=row, j=col, k=slice) → patient (x, y, z)
    #   axis 0 (i, rows)  : step = col_cos * row_spacing
    #   axis 1 (j, cols)  : step = row_cos * col_spacing
    #   axis 2 (k, slices): step = slice_vec
    affine = np.eye(4)
    affine[:3, 0] = col_cos * ps[0]   # row spacing along row axis
    affine[:3, 1] = row_cos * ps[1]   # col spacing along col axis
    affine[:3, 2] = slice_vec
    affine[:3, 3] = ipp0

    nifti = nib.Nifti1Image(volume, affine)
    return volume, nifti


# ── Visualisation ─────────────────────────────────────────────────────────────

def pick_colormap(n_labels: int) -> ListedColormap:
    """Random distinct colours for segmentation labels."""
    rng = np.random.default_rng(42)
    colours = np.zeros((n_labels + 1, 4))
    colours[0] = [0, 0, 0, 0]          # label 0 → transparent
    colours[1:] = rng.uniform(0.3, 1.0, size=(n_labels, 4))
    colours[1:, 3] = 0.55              # alpha
    return ListedColormap(colours)


def show_segmentations(volume: np.ndarray, seg_nifti: nib.Nifti1Image,
                       study_name: str, n_slices: int = 6,
                       out_path: str = "segmentation_overlay.png"):
    """Display n_slices evenly-spaced axial slices with segmentation overlay."""
    seg = seg_nifti.get_fdata().astype(np.int32)   # (H, W, Z)

    n_z = volume.shape[2]
    indices = np.linspace(n_z // 5, 4 * n_z // 5, n_slices, dtype=int)

    n_labels = int(seg.max())
    cmap = pick_colormap(n_labels)

    fig, axes = plt.subplots(2, n_slices // 2, figsize=(18, 8))
    axes = axes.flatten()
    fig.suptitle(f"TotalSegmentator — {study_name}", fontsize=13)

    for ax, idx in zip(axes, indices):
        ct_slice  = volume[:, :, idx]
        seg_slice = seg[:, :, idx]

        # CT in grayscale (brain window), origin="upper" = row 0 at top (standard CT)
        ax.imshow(ct_slice, cmap="gray", vmin=-40, vmax=80)

        # Segmentation overlay
        seg_rgba = cmap(seg_slice.astype(float) / max(n_labels, 1))
        seg_rgba[seg_slice == 0, 3] = 0   # background transparent
        ax.imshow(seg_rgba)

        ax.set_title(f"Slice {idx}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    import subprocess, sys
    subprocess.run(["open", out_path], check=False)   # macOS: open in Preview


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TotalSegmentator on demo CT study")
    parser.add_argument(
        "--study",
        default="demo_studies/positive/subarachnoid__ID_9fae411ae8",
        help="Path to study folder containing .dcm files",
    )
    parser.add_argument(
        "--task",
        default="total_mr_highres",
        help="TotalSegmentator task (default: total_mr_highres; try 'total' for CT)",
    )
    parser.add_argument("--slices", type=int, default=6, help="Number of slices to display")
    args = parser.parse_args()

    study_path = Path(args.study)
    if not study_path.exists():
        raise SystemExit(f"Study folder not found: {study_path}")

    print(f"Loading DICOM series from: {study_path}")
    volume, input_nifti = load_dicom_series(str(study_path))
    print(f"  Volume shape: {volume.shape}  dtype: {volume.dtype}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path  = os.path.join(tmpdir, "input.nii.gz")
        output_path = os.path.join(tmpdir, "segmentation")

        print("Saving NIfTI...")
        nib.save(input_nifti, input_path)

        print(f"Running TotalSegmentator (task={args.task}) — this may take a few minutes...")
        from totalsegmentator.python_api import totalsegmentator
        seg_nifti = totalsegmentator(input_path, output_path, task=args.task,
                                     ml=True, verbose=True)

        # totalsegmentator may return a path or NIfTI object
        if isinstance(seg_nifti, (str, Path)):
            seg_nifti = nib.load(seg_nifti)

        out_png = f"segmentation_{study_path.name}.png"
        print("Rendering segmentations...")
        show_segmentations(volume, seg_nifti, study_path.name,
                           n_slices=args.slices, out_path=out_png)


if __name__ == "__main__":
    main()
