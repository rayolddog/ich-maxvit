"""
Robust DICOM → HU reader for 1-channel CT pipeline.

Handles:
  - Big-endian transfer syntax (Explicit VR Big Endian: 1.2.840.10008.1.2.2)
    pydicom manages byte-order automatically; manual fallback included for
    pathological files where pixel_array() fails.
  - BitsAllocated: 8, 16, 32, 64
  - BitsStored: 8, 10, 12, 16  (pydicom handles 12-bit packing transparently)
  - PixelRepresentation: 0 (unsigned) or 1 (signed)
  - Float pixel data (FloatPixelData / DoubleFloatPixelData DICOM tags,
    BitsAllocated 32/64 floats)
  - Missing or present RescaleSlope / RescaleIntercept

Exclusion rules:
  1. pydicom.dcmread raises an exception
  2. pixel_array cannot be decoded or result is not 2D
  3. BitsAllocated absent or not in {8, 16, 32, 64}
  4. HU range implausible after conversion:
       min > 200             → raw un-converted pixel values (e.g. uint16 0-65535)
       min >= 0, max <= 400  → un-converted unsigned pixels
       max > 4096, min >= 0  → intercept likely not applied (all-positive with
                               very high values; metal is always accompanied by
                               negative air pixels)
       max < 0               → no tissue present (scout/localizer/empty slice)

HU conversion priority:
  1. RescaleSlope / RescaleIntercept present → HU = pixel * slope + intercept
  2. Float pixel data, no rescale → assume already in physical/HU units
  3. uint16, no intercept → HU = pixel − 1024  (common scanner default)
  4. int16, int8, uint8, no intercept → treat as direct HU; rule 4 validates
"""

import numpy as np
from pathlib import Path

# Transfer Syntax UIDs
_BIG_ENDIAN_TS    = '1.2.840.10008.1.2.2'   # Explicit VR Big Endian
_IMPL_LE_TS       = '1.2.840.10008.1.2'      # Implicit VR Little Endian
_EXPL_LE_TS       = '1.2.840.10008.1.2.1'    # Explicit VR Little Endian

SUPPORTED_BITS = {8, 16, 32, 64}


def extract_image_id(file_path: str) -> str:
    """Extract image ID from a DICOM filename.

    Kaggle RSNA format: 'ID_xxxxxxxxx.dcm' → 'ID_xxxxxxxxx'.
    Other UIDs: extracts last alphanumeric component.
    """
    stem = Path(file_path).stem

    if stem.startswith('ID_'):
        return stem

    for sep in ['.', '_', '-']:
        if sep in stem:
            last_part = stem.split(sep)[-1]
            if last_part.isdigit() or last_part.isalnum():
                return f"ID_{last_part}"

    safe_name = "".join(c if c.isalnum() or c in '-_' else '_' for c in stem)
    return safe_name


def _big_endian_pixel_fallback(ds) -> np.ndarray:
    """Manually byte-swap raw pixel data when pydicom's normal path fails
    on big-endian files.

    pydicom 3.x usually handles big-endian automatically, but some files
    with non-standard headers need this fallback.
    """
    bits = getattr(ds, 'BitsAllocated', 16)
    rows = getattr(ds, 'Rows')
    cols = getattr(ds, 'Columns')
    pr   = getattr(ds, 'PixelRepresentation', 0)   # 0=unsigned, 1=signed
    raw  = bytes(ds.PixelData)

    if bits == 8:
        dtype = np.int8 if pr == 1 else np.uint8
        arr = np.frombuffer(raw, dtype=dtype)
    elif bits == 16:
        # Read as big-endian uint16, convert to native, re-view if signed
        arr = np.frombuffer(raw, dtype=np.dtype('>u2')).astype(np.uint16)
        if pr == 1:
            arr = arr.view(np.int16)
    elif bits == 32:
        # Check if float (some float32 CT data with BitsAllocated=32)
        arr = np.frombuffer(raw, dtype=np.dtype('>u4')).astype(np.uint32)
        if pr == 1:
            arr = arr.view(np.int32)
    elif bits == 64:
        arr = np.frombuffer(raw, dtype=np.dtype('>u8')).astype(np.uint64)
        if pr == 1:
            arr = arr.view(np.int64)
    else:
        raise ValueError(f"BitsAllocated={bits} not supported in big-endian fallback")

    return arr.reshape(rows, cols)


def read_dicom_hu(file_path: str):
    """Read a DICOM file and return a float32 HU array.

    Args:
        file_path: Path to DICOM file.

    Returns:
        Tuple (hu_array, image_id, is_valid):
            hu_array: float32 ndarray (H, W) in Hounsfield Units, or None.
            image_id: str — extracted from filename.
            is_valid: bool — False if any exclusion rule triggers.
    """
    import pydicom
    import warnings

    image_id = extract_image_id(file_path)

    # Rule 1: must be readable (force=True accepts files without DICOM preamble)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ds = pydicom.dcmread(file_path, force=True)
    except Exception:
        return None, image_id, False

    # Rule 3: bit depth must be known and supported
    bits = getattr(ds, 'BitsAllocated', None)
    if bits not in SUPPORTED_BITS:
        return None, image_id, False

    # Determine transfer syntax (big-endian detection)
    ts = ''
    if hasattr(ds, 'file_meta') and hasattr(ds.file_meta, 'TransferSyntaxUID'):
        ts = str(ds.file_meta.TransferSyntaxUID)
    is_big_endian = (ts == _BIG_ENDIAN_TS)

    # Rule 2: pixel array must be decodable and 2-dimensional
    pixel_array = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            pixel_array = ds.pixel_array
    except Exception:
        if is_big_endian:
            # Manual fallback for big-endian files where pydicom's handler fails
            try:
                pixel_array = _big_endian_pixel_fallback(ds)
            except Exception:
                return None, image_id, False
        else:
            return None, image_id, False

    if pixel_array is None or pixel_array.ndim != 2:
        return None, image_id, False

    # Detect float pixel data (BitsAllocated 32 or 64, dtype float)
    is_float = pixel_array.dtype.kind == 'f'

    pixel_f32 = pixel_array.astype(np.float32)

    # HU conversion
    has_intercept = hasattr(ds, 'RescaleIntercept')
    has_slope     = hasattr(ds, 'RescaleSlope')

    if has_intercept or has_slope:
        # Standard path: HU = pixel * slope + intercept
        slope     = float(getattr(ds, 'RescaleSlope', 1.0))
        intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
        hu = pixel_f32 * slope + intercept

    elif is_float:
        # Float pixels without rescale tags: assume physical units ≈ HU
        hu = pixel_f32

    elif pixel_array.dtype == np.uint16:
        # uint16, no intercept: scanner-default offset convention
        hu = pixel_f32 - 1024.0

    else:
        # int16, int8, uint8, etc. without intercept:
        # some scanners store HU directly; rule 4 will reject implausible values
        hu = pixel_f32

    # Rule 4: implausible HU range
    hu_min = float(hu.min())
    hu_max = float(hu.max())

    # All values > 200 → raw un-converted pixels (e.g. uint16 0-65535)
    if hu_min > 200:
        return None, image_id, False

    # All values in [0, 400]: unsigned pixels, no conversion applied
    if hu_min >= 0 and hu_max <= 400:
        return None, image_id, False

    # Values > 4096 with no negative pixels → intercept likely not applied
    # (valid metal/hardware can push HU above 3000 but always with negative
    #  air pixels present; if hu_min >= 0 the whole image is positive which
    #  is only possible if the intercept was skipped)
    if hu_max > 4096 and hu_min >= 0:
        return None, image_id, False

    # hu_max < 0 → no tissue at all (scout image, localizer, or empty slice)
    if hu_max < 0:
        return None, image_id, False

    return hu.astype(np.float32), image_id, True
