# Hemorrhage annotation assist (personal preprocessing for ITK-SNAP / 3D Slicer)

Turns one NCCT head study into domain-specific inputs that make the approved
annotation tools fast for acute hemorrhage. Introduces no new software — emits
standard geometry-preserving NIfTI the tools already ingest.

```
.venv/bin/python annotation_assist/bloodness_speed.py <dicom_series_dir> \
      --out annotation_assist/out/<name>
.venv/bin/python annotation_assist/bloodness_speed.py --selftest
```

Outputs: `_speed.nii.gz` (ITK-SNAP active-contour feature), `_seed.nii.gz`
(3D Slicer segmentation seed; 1=blood candidate, 2=bone), `_QC.png`
(axial/coronal/sagittal check through the largest component).

Method (all in calibrated HU, no windowing/saturation):
1. acute-blood membership, plateau 52-72 HU (above gray matter to avoid cortex);
2. intracranial mask (fill inside the skull) — excludes extracranial muscle
   (also ~40-60 HU), the dominant contaminant;
3. thin-rim removal by per-slice morphological opening — deletes the ~1-voxel
   inner-table partial-volume rim while keeping thick collections that hug bone
   (thickness, not distance, is the discriminator — real SDH touches bone too).

KNOWN LIMITS (v1): targets ACUTE (hyperdense) blood only; residual specificity
loss to cortical vessels / gray-matter PV at the surface (next: local-contrast
top-hat); coronal/sagittal reformats are only as good as slice thickness — 5 mm
data gives coarse reformats (subfrontal/subtemporal SDH need thin-slice source);
GDCM decodes JPEG-Lossless CQ500 natively. Requires the repo venv (SimpleITK,
scipy, matplotlib).
