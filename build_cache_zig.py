"""
Build CT HU tensor caches using the Zig shared library for fast windowing.

Pipeline per DICOM file:
  1. Python / pydicom  — read DICOM, extract pixel_array, apply RescaleSlope +
                         RescaleIntercept, validate HU range (dicom_reader_1ch)
  2. Zig (libhu_tensor) — apply fixed HU window → float16 tensor (u16 bit patterns)
  3. Python / numpy    — reshape, save as .npz

All three window types (wide / medium / narrow) can be built in a single DICOM
pass using apply_three_windows(), which reads the HU array once and writes three
output tensors.  Use --window all for this mode.

Defaults build the medium window only.  Override with --window wide|medium|narrow|all.

Output directories (configurable):
  --train-dir-wide    /home/johnb/cache_wide_train
  --test-dir-wide     /home/johnb/cache_wide_test
  --train-dir-medium  /home/johnb/cache_medium_train
  --test-dir-medium   /home/johnb/cache_medium_test
  --train-dir-narrow  /home/johnb/cache_narrow_train   (already built; skip if exists)
  --test-dir-narrow   /home/johnb/cache_narrow_test
"""

import os
import sys
import json
import time
import ctypes
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

NEWICH_DIR   = os.path.dirname(os.path.abspath(__file__))
LIB_DEFAULT  = os.path.join(NEWICH_DIR, 'hu_tensor', 'zig-out', 'lib', 'libhu_tensor.so')
SPLITS_DEFAULT = os.path.join(NEWICH_DIR, 'checkpoints_1ch', 'data_splits.json')
DCM_DEFAULT  = '/home/johnb/DCMdata'
TARGET_SIZE  = 512

# ── HU window parameters (must match hu_windows.py) ──────────────────────────
WINDOWS = {
    'wide'  : (-1024.0, 3071.0),
    'medium': ( -200.0,  200.0),
    'narrow': (   48.0,   90.0),
}

# ── Zig library interface ─────────────────────────────────────────────────────

def _load_lib(lib_path: str) -> ctypes.CDLL:
    """Load libhu_tensor.so and set up function signatures."""
    lib = ctypes.CDLL(lib_path)

    f32_ptr = ctypes.POINTER(ctypes.c_float)
    u16_ptr = ctypes.POINTER(ctypes.c_uint16)

    # apply_window(hu, n, hu_low, hu_high, output)
    lib.apply_window.restype  = None
    lib.apply_window.argtypes = [
        f32_ptr,        # hu
        ctypes.c_size_t,# n
        ctypes.c_float, # hu_low
        ctypes.c_float, # hu_high
        u16_ptr,        # output (f16 bit patterns)
    ]

    # apply_three_windows(hu, n, lo0,hi0, lo1,hi1, lo2,hi2, out0, out1, out2)
    lib.apply_three_windows.restype  = None
    lib.apply_three_windows.argtypes = [
        f32_ptr, ctypes.c_size_t,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float,
        ctypes.c_float, ctypes.c_float,
        u16_ptr, u16_ptr, u16_ptr,
    ]

    # Version check
    lib.hu_tensor_version.restype  = ctypes.c_uint32
    lib.hu_tensor_version.argtypes = []

    return lib


def _call_apply_window(lib, hu: np.ndarray,
                       hu_low: float, hu_high: float) -> np.ndarray:
    """Call Zig apply_window; return float16 ndarray (512, 512)."""
    hu_c  = np.ascontiguousarray(hu, dtype=np.float32)
    out_u = np.empty(hu_c.size, dtype=np.uint16)
    lib.apply_window(
        hu_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(hu_c.size),
        ctypes.c_float(hu_low),
        ctypes.c_float(hu_high),
        out_u.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return out_u.view(np.float16).reshape(hu.shape)


def _call_apply_three_windows(lib, hu: np.ndarray,
                               params: list[tuple]) -> list[np.ndarray]:
    """
    Call Zig apply_three_windows; params is list of (hu_low, hu_high) triples.
    Returns list of three float16 ndarrays (512, 512).
    """
    hu_c = np.ascontiguousarray(hu, dtype=np.float32)
    n    = hu_c.size
    outs = [np.empty(n, dtype=np.uint16) for _ in range(3)]
    ptr  = ctypes.POINTER(ctypes.c_uint16)

    lib.apply_three_windows(
        hu_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(n),
        ctypes.c_float(params[0][0]), ctypes.c_float(params[0][1]),
        ctypes.c_float(params[1][0]), ctypes.c_float(params[1][1]),
        ctypes.c_float(params[2][0]), ctypes.c_float(params[2][1]),
        outs[0].ctypes.data_as(ptr),
        outs[1].ctypes.data_as(ptr),
        outs[2].ctypes.data_as(ptr),
    )
    shape = hu.shape
    return [o.view(np.float16).reshape(shape) for o in outs]


# ── Per-worker state (loaded once per worker process via initializer) ─────────

_worker_lib       = None
_worker_lib_path  = None
_worker_mode      = None   # 'single' or 'all'
_worker_win_name  = None   # 'wide'|'medium'|'narrow' for single mode
_worker_allowed   = None   # frozenset of image_ids for this split

def _worker_init(lib_path: str, mode: str, win_name: str, allowed_ids):
    global _worker_lib, _worker_lib_path, _worker_mode
    global _worker_win_name, _worker_allowed
    _worker_lib_path = lib_path
    _worker_lib      = _load_lib(lib_path)
    _worker_mode     = mode
    _worker_win_name = win_name
    _worker_allowed  = allowed_ids


# ── Worker function ───────────────────────────────────────────────────────────

def _process_one(args_tuple):
    """
    Worker: process one DICOM file.

    args_tuple: (file_path, out_dirs_dict)
      out_dirs_dict maps window name → output directory path.

    Returns: (image_id, status_str, error_msg_or_None)
      status: 'ok' | 'skip' | 'invalid' | 'error'
    """
    file_path, out_dirs = args_tuple

    sys.path.insert(0, NEWICH_DIR)
    from dicom_reader_1ch import read_dicom_hu, extract_image_id

    image_id = extract_image_id(file_path)

    # Split membership check
    if _worker_allowed is not None and image_id not in _worker_allowed:
        return image_id, 'skip', None

    # Check if all outputs already exist (resume)
    all_exist = all(
        os.path.exists(os.path.join(d, f"{image_id}.npz"))
        for d in out_dirs.values()
    )
    if all_exist:
        return image_id, 'skip', None

    # DICOM → validated float32 HU
    try:
        hu, image_id, is_valid = read_dicom_hu(file_path)
    except Exception as e:
        return image_id, 'error', f"read_dicom_hu: {e}"

    if not is_valid or hu is None:
        return image_id, 'invalid', 'excluded by HU validation rules'

    # Resize to TARGET_SIZE × TARGET_SIZE if needed
    h, w = hu.shape
    if h != TARGET_SIZE or w != TARGET_SIZE:
        try:
            from scipy.ndimage import zoom
            hu = zoom(hu, (TARGET_SIZE / h, TARGET_SIZE / w), order=1)
        except Exception as e:
            return image_id, 'error', f"resize: {e}"

    # Apply windows via Zig
    try:
        if _worker_mode == 'all':
            win_order = ['wide', 'medium', 'narrow']
            params    = [WINDOWS[w] for w in win_order]
            tensors   = _call_apply_three_windows(_worker_lib, hu, params)
            results   = dict(zip(win_order, tensors))
        else:
            lo, hi   = WINDOWS[_worker_win_name]
            tensor   = _call_apply_window(_worker_lib, hu, lo, hi)
            results  = {_worker_win_name: tensor}
    except Exception as e:
        return image_id, 'error', f"Zig apply_window: {e}"

    # Save outputs
    for win_name, tensor in results.items():
        out_dir  = out_dirs[win_name]
        out_path = os.path.join(out_dir, f"{image_id}.npz")
        if os.path.exists(out_path):
            continue
        try:
            np.savez_compressed(out_path, image_norm=tensor)
        except Exception as e:
            return image_id, 'error', f"save {win_name}: {e}"

    return image_id, 'ok', None


# ── Cache builder ─────────────────────────────────────────────────────────────

def build_cache(dcm_dir: str, out_dirs: dict, allowed_ids,
                label: str, lib_path: str, mode: str, win_name: str,
                workers: int, batch_size: int):
    """
    Process DICOMs for one split (train+val or test).

    out_dirs: dict mapping window_name → output directory
    """
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Log file: shared across windows for this split
    log_path = os.path.join(next(iter(out_dirs.values())), 'BadDicomFiles.log')

    # Resume: union of already-cached IDs across all output dirs
    existing = set()
    for d in out_dirs.values():
        with os.scandir(d) as it:
            for entry in it:
                if entry.name.endswith('.npz') and entry.is_file():
                    existing.add(entry.name[:-4])

    print(f"\n[{label}] Resume: {len(existing):,} already cached")

    # Collect DICOM paths
    all_files = []
    for dirpath, _, filenames in os.walk(dcm_dir):
        for fname in filenames:
            if fname.lower().endswith('.dcm'):
                stem = Path(fname).stem
                if stem in existing:
                    continue
                all_files.append(os.path.join(dirpath, fname))

    total = len(all_files)
    print(f"[{label}] Files to process: {total:,}")

    written  = 0
    skipped  = 0
    invalid  = 0
    errors   = 0
    t0       = time.perf_counter()

    init_args = (lib_path, mode, win_name, frozenset(allowed_ids))

    with open(log_path, 'a') as log_fh:
        with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=init_args) as executor:

            offset = 0
            while offset < total:
                batch = all_files[offset:offset + batch_size]
                offset += batch_size

                futures = {
                    executor.submit(_process_one, (fp, out_dirs)): fp
                    for fp in batch
                }

                for fut in as_completed(futures):
                    fp = futures[fut]
                    try:
                        image_id, status, errmsg = fut.result()
                    except Exception as exc:
                        ts = datetime.now().isoformat(timespec='seconds')
                        log_fh.write(f"{ts}\t{fp}\texecutor exception: {exc}\n")
                        errors += 1
                        continue

                    if status == 'ok':
                        written += 1
                    elif status == 'skip':
                        skipped += 1
                    elif status == 'invalid':
                        invalid += 1
                    else:
                        errors += 1
                        ts = datetime.now().isoformat(timespec='seconds')
                        log_fh.write(f"{ts}\t{fp}\t{errmsg}\n")

                    if (written + errors) % 10000 == 0 and (written + errors) > 0:
                        elapsed = time.perf_counter() - t0
                        rate    = written / elapsed if elapsed > 0 else 0
                        remain  = len(allowed_ids) - written
                        eta     = remain / rate if rate > 0 else 0
                        print(f"  [{label}] {written:,} written  "
                              f"{invalid:,} invalid  {errors:,} errors  "
                              f"{rate:.0f}/s  ETA {eta/60:.1f} min",
                              flush=True)

    elapsed = time.perf_counter() - t0
    print(f"[{label}] Done: {written:,} written, {skipped:,} skipped, "
          f"{invalid:,} invalid, {errors:,} errors  ({elapsed/60:.1f} min)")
    return errors


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build CT HU tensor cache via Zig windowing library')
    parser.add_argument('--dcm-dir',     default=DCM_DEFAULT)
    parser.add_argument('--splits-file', default=SPLITS_DEFAULT)
    parser.add_argument('--lib',         default=LIB_DEFAULT,
                        help='Path to libhu_tensor.so')
    parser.add_argument('--window',      default='medium',
                        choices=['wide', 'medium', 'narrow', 'all'],
                        help='Which HU window(s) to build (default: medium)')
    parser.add_argument('--train-dir-wide',   default='/home/johnb/cache_wide_train')
    parser.add_argument('--test-dir-wide',    default='/home/johnb/cache_wide_test')
    parser.add_argument('--train-dir-medium', default='/home/johnb/cache_medium_train')
    parser.add_argument('--test-dir-medium',  default='/home/johnb/cache_medium_test')
    parser.add_argument('--train-dir-narrow', default='/home/johnb/cache_narrow_train')
    parser.add_argument('--test-dir-narrow',  default='/home/johnb/cache_narrow_test')
    parser.add_argument('--workers',     type=int, default=8)
    parser.add_argument('--batch-size',  type=int, default=1000)
    args = parser.parse_args()

    # Verify library loads and version check
    lib = _load_lib(args.lib)
    ver = lib.hu_tensor_version()
    print(f"Loaded {args.lib}  version {ver // 10000}.{(ver // 100) % 100}.{ver % 100}")

    # Determine which windows to build
    if args.window == 'all':
        active_wins = ['wide', 'medium', 'narrow']
        mode = 'all'
    else:
        active_wins = [args.window]
        mode = 'single'

    dir_map = {
        'wide'  : {'train': args.train_dir_wide,   'test': args.test_dir_wide},
        'medium': {'train': args.train_dir_medium,  'test': args.test_dir_medium},
        'narrow': {'train': args.train_dir_narrow,  'test': args.test_dir_narrow},
    }

    print(f"Window(s)   : {active_wins}")
    for w in active_wins:
        lo, hi = WINDOWS[w]
        print(f"  {w:8s}: {lo:.0f} to {hi:.0f} HU  "
              f"train→{dir_map[w]['train']}  test→{dir_map[w]['test']}")
    print(f"DICOM source: {args.dcm_dir}")
    print(f"Splits      : {args.splits_file}")
    print(f"Workers     : {args.workers}")

    with open(args.splits_file) as f:
        splits = json.load(f)

    train_ids    = set(splits['train_ids'])
    val_ids      = set(splits['val_ids'])
    test_ids     = set(splits['test_ids'])
    trainval_ids = train_ids | val_ids

    print(f"\n  train+val : {len(trainval_ids):,}")
    print(f"  test      : {len(test_ids):,}")

    # Build output dir dicts for each split
    train_out_dirs = {w: dir_map[w]['train'] for w in active_wins}
    test_out_dirs  = {w: dir_map[w]['test']  for w in active_wins}

    t_start = time.perf_counter()

    errs1 = build_cache(args.dcm_dir, train_out_dirs, trainval_ids,
                        'TRAIN+VAL', args.lib, mode, args.window,
                        args.workers, args.batch_size)
    errs2 = build_cache(args.dcm_dir, test_out_dirs, test_ids,
                        'TEST', args.lib, mode, args.window,
                        args.workers, args.batch_size)

    total_elapsed = time.perf_counter() - t_start
    print(f"\nAll done in {total_elapsed/60:.1f} min  "
          f"(total bad files: {errs1 + errs2})")


if __name__ == '__main__':
    main()
