"""
Build medium-window HU tensor cache directly from DICOM files.

HU window: −200 to 200 HU → [0.0, 1.0]  (see hu_windows.WINDOW_MEDIUM)

Reads DICOM files from --dcm-dir (recursive scan).
Splits output using the same train/val/test split as the narrow cache:
  TRAIN_DIR  ← train_ids + val_ids   (used during training runs)
  TEST_DIR   ← test_ids              (held out)

Each output .npz contains:
  image_norm : float16 (512, 512)  — medium-HU mapped values in [0.0, 1.0]

Resume logic: files already present in the output dir are skipped.
Bad files: logged to BadDicomFiles.log in each output dir.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

NEWICH_DIR = os.path.dirname(os.path.abspath(__file__))

DCM_DIR_DEFAULT   = '/home/johnb/DCMdata'
TRAIN_DIR_DEFAULT = '/home/johnb/cache_medium_train'
TEST_DIR_DEFAULT  = '/home/johnb/cache_medium_test'
SPLITS_FILE_DEFAULT = '/home/johnb/NewICH/checkpoints_1ch/data_splits.json'

TARGET_SIZE = 512


# ── Worker ────────────────────────────────────────────────────────────────────

def _process_one(args_tuple):
    """Worker: DICOM path → (image_id, float16 tensor, error_str | None)."""
    file_path, = args_tuple

    sys.path.insert(0, NEWICH_DIR)
    from dicom_reader_1ch import read_dicom_hu, extract_image_id
    from hu_windows import apply_window, WINDOW_MEDIUM

    image_id = extract_image_id(file_path)

    try:
        hu, image_id, is_valid = read_dicom_hu(file_path)
    except Exception as e:
        return image_id, None, f"read_dicom_hu exception: {e}"

    if not is_valid or hu is None:
        return image_id, None, "invalid DICOM (excluded by HU validation rules)"

    # Resize to 512×512 if needed
    h, w = hu.shape
    if h != TARGET_SIZE or w != TARGET_SIZE:
        try:
            from scipy.ndimage import zoom
            hu = zoom(hu, (TARGET_SIZE / h, TARGET_SIZE / w), order=1)
        except Exception as e:
            return image_id, None, f"resize error: {e}"

    tensor = apply_window(hu, WINDOW_MEDIUM)
    return image_id, tensor, None


# ── Cache builder ─────────────────────────────────────────────────────────────

def scan_dcm_files(dcm_dir: str):
    """Recursively yield all .dcm file paths under dcm_dir."""
    for dirpath, _, filenames in os.walk(dcm_dir):
        for fname in filenames:
            if fname.lower().endswith('.dcm'):
                yield os.path.join(dirpath, fname)


def build_split_cache(dcm_dir: str, output_dir: str, allowed_ids: set,
                      label: str, workers: int, batch_size: int):
    """Process all DICOMs whose image_id is in allowed_ids; write to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, 'BadDicomFiles.log')

    # Resume: collect already-cached IDs
    existing_ids = set()
    with os.scandir(output_dir) as it:
        for entry in it:
            if entry.name.endswith('.npz') and entry.is_file():
                existing_ids.add(entry.name[:-4])
    print(f"  [{label}] Resume set: {len(existing_ids):,} already cached")

    # Collect DICOM paths (quick stem filter against existing + allowed sets)
    all_files = []
    for fpath in scan_dcm_files(dcm_dir):
        stem = Path(fpath).stem
        if stem in existing_ids:
            continue
        # Include file if stem matches allowed_ids OR if we can't tell yet
        # (real image_id is extracted by the worker for non-standard names)
        if stem in allowed_ids or stem.startswith('ID_'):
            all_files.append(fpath)

    print(f"  [{label}] Files to process: {len(all_files):,}  → {output_dir}")

    written = 0
    skipped = 0
    bad     = 0
    t0      = time.perf_counter()
    last_milestone = 0
    total   = len(all_files)

    with open(log_path, 'a') as log_fh:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            offset = 0
            while offset < total:
                batch = all_files[offset:offset + batch_size]
                offset += batch_size

                futures = {executor.submit(_process_one, (fp,)): fp
                           for fp in batch}

                for fut in as_completed(futures):
                    fp = futures[fut]
                    try:
                        image_id, tensor, error = fut.result()
                    except Exception as exc:
                        ts = datetime.now().isoformat(timespec='seconds')
                        log_fh.write(f"{ts}\t{fp}\texecutor exception: {exc}\n")
                        bad += 1
                        continue

                    if error is not None:
                        ts = datetime.now().isoformat(timespec='seconds')
                        log_fh.write(f"{ts}\t{fp}\t{error}\n")
                        bad += 1
                        continue

                    # Drop IDs not in this split's allowed set
                    if image_id not in allowed_ids:
                        skipped += 1
                        continue

                    # Precise resume check (worker may return real image_id)
                    if image_id in existing_ids:
                        skipped += 1
                        continue

                    out_path = os.path.join(output_dir, f"{image_id}.npz")
                    try:
                        np.savez_compressed(out_path, image_norm=tensor)
                        existing_ids.add(image_id)
                        written += 1
                    except Exception as exc:
                        ts = datetime.now().isoformat(timespec='seconds')
                        log_fh.write(f"{ts}\t{fp}\tsave error: {exc}\n")
                        bad += 1
                        continue

                    milestone = written // 10000
                    if milestone > last_milestone:
                        last_milestone = milestone
                        elapsed = time.perf_counter() - t0
                        rate = written / elapsed if elapsed > 0 else 0
                        eta = (len(allowed_ids) - written) / rate if rate > 0 else 0
                        print(f"  [{label}] {written:,} written  "
                              f"({bad} bad, {skipped} skipped)  "
                              f"{rate:.0f}/s  ETA {eta/60:.1f} min",
                              flush=True)

    elapsed = time.perf_counter() - t0
    print(f"  [{label}] Done: {written:,} written, "
          f"{skipped:,} skipped, {bad:,} bad  "
          f"({elapsed/60:.1f} min)")
    return bad


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Build medium-window HU cache (−200 to 200 HU → [0, 1])')
    parser.add_argument('--dcm-dir',     default=DCM_DIR_DEFAULT)
    parser.add_argument('--train-dir',   default=TRAIN_DIR_DEFAULT)
    parser.add_argument('--test-dir',    default=TEST_DIR_DEFAULT)
    parser.add_argument('--splits-file', default=SPLITS_FILE_DEFAULT)
    parser.add_argument('--workers',     type=int, default=8)
    parser.add_argument('--batch-size',  type=int, default=1000)
    args = parser.parse_args()

    # Import here so the docstring is visible at module level without side effects
    sys.path.insert(0, NEWICH_DIR)
    from hu_windows import WINDOW_MEDIUM
    print(f"Medium-window HU cache builder")
    print(f"  Window      : {WINDOW_MEDIUM}")
    print(f"  DICOM source: {args.dcm_dir}")
    print(f"  Train dir   : {args.train_dir}")
    print(f"  Test dir    : {args.test_dir}")
    print(f"  Splits      : {args.splits_file}")
    print(f"  Workers     : {args.workers}")

    with open(args.splits_file) as f:
        splits = json.load(f)

    train_ids    = set(splits['train_ids'])
    val_ids      = set(splits['val_ids'])
    test_ids     = set(splits['test_ids'])
    trainval_ids = train_ids | val_ids

    print(f"\n  train+val : {len(trainval_ids):,}")
    print(f"  test      : {len(test_ids):,}")
    print(f"  total     : {len(trainval_ids) + len(test_ids):,}\n")

    t_start = time.perf_counter()

    errs1 = build_split_cache(args.dcm_dir, args.train_dir, trainval_ids,
                               'TRAIN+VAL', args.workers, args.batch_size)
    errs2 = build_split_cache(args.dcm_dir, args.test_dir, test_ids,
                               'TEST', args.workers, args.batch_size)

    total_elapsed = time.perf_counter() - t_start
    print(f"\nAll done in {total_elapsed/60:.1f} min  "
          f"(total bad files: {errs1 + errs2})")


if __name__ == '__main__':
    main()
