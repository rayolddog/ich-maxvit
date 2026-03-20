"""
Dataset for 1-channel CT .npz tensors (pure classification, no segmentation).

.npz file key:
  image_norm  float16 (512, 512): normalized HU in [0, 1]

Each __getitem__ returns:
  image_1ch:  Tensor (1, 512, 512) float32
  cls_labels: Tensor (6,) float32

Geometric augmentation: random horizontal flip + rot90.

Label utilities (load_labels, multilabel_stratified_split) reuse the same
CSV format and logic as ClaudeCT/dataset.py.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from scipy import ndimage
from typing import List, Tuple, Optional

# ── Label definitions (same as ClaudeCT/dataset.py) ──────────────────────────
LABEL_COLS = ['epidural', 'intraparenchymal', 'intraventricular',
              'subarachnoid', 'subdural', 'any']


# ── Label utilities ───────────────────────────────────────────────────────────

def load_labels(csv_path: str, available_ids: set) -> pd.DataFrame:
    """Load and pivot the RSNA labels CSV to one row per image.

    Args:
        csv_path: Path to stage_2_train.csv.
        available_ids: Set of image IDs that have cached .npz files.

    Returns:
        DataFrame indexed by image_id with columns = LABEL_COLS (int 0/1).
    """
    print("Loading labels...")
    df = pd.read_csv(csv_path)

    df['image_id'] = df['ID'].apply(lambda x: '_'.join(x.split('_')[:2]))
    df['label_type'] = df['ID'].apply(lambda x: x.split('_')[2])

    df = df[df['image_id'].isin(available_ids)]
    df = df.drop_duplicates(subset=['image_id', 'label_type'], keep='first')

    labels_pivot = df.pivot(index='image_id', columns='label_type', values='Label')
    labels_pivot = labels_pivot[LABEL_COLS]
    labels_pivot = labels_pivot.fillna(0).astype(int)

    print(f"Loaded labels for {len(labels_pivot)} images")
    print("\nClass distribution:")
    for col in LABEL_COLS:
        pos = labels_pivot[col].sum()
        total = len(labels_pivot)
        print(f"  {col}: {pos}/{total} ({100 * pos / total:.2f}%)")

    return labels_pivot


def multilabel_stratified_split(
        image_ids: List[str], labels_df: pd.DataFrame,
        val_split: float, random_state: int = 42
) -> Tuple[List[str], List[str]]:
    """Stratified train/val split preserving per-label positive rates.

    Groups images by their full 6-label binary signature so all combinations
    (including rare subtypes) are proportionally represented in both splits.
    Groups with a single sample go entirely to train.
    """
    rng = np.random.RandomState(random_state)
    labels = labels_df.loc[image_ids, LABEL_COLS].values

    groups: dict = {}
    for i, img_id in enumerate(image_ids):
        key = tuple(int(v) for v in labels[i])
        groups.setdefault(key, []).append(img_id)

    train_ids: List[str] = []
    val_ids: List[str] = []
    for ids in groups.values():
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * val_split)) if len(ids) >= 2 else 0
        val_ids.extend(ids[:n_val])
        train_ids.extend(ids[n_val:])

    return train_ids, val_ids


def split_three_way(
        image_ids: List[str], labels_df: pd.DataFrame,
        n_test: int, val_split: float = 0.15, random_state: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Stratified three-way split: test / train / val.

    Draws n_test samples proportionally from each label-signature group for
    the held-out test set, then applies multilabel_stratified_split on the
    remainder for train/val.

    Args:
        image_ids:    All candidate image IDs.
        labels_df:    DataFrame indexed by image_id with LABEL_COLS columns.
        n_test:       Number of images to reserve for the final test set.
        val_split:    Fraction of the non-test pool to use for validation.
        random_state: RNG seed (reproducible splits).

    Returns:
        (test_ids, train_ids, val_ids)
    """
    rng = np.random.RandomState(random_state)
    labels = labels_df.loc[image_ids, LABEL_COLS].values

    # Group by 6-bit label signature
    groups: dict = {}
    for i, img_id in enumerate(image_ids):
        key = tuple(int(v) for v in labels[i])
        groups.setdefault(key, []).append(img_id)

    total = len(image_ids)
    test_fraction = n_test / total

    test_ids: List[str] = []
    remainder: List[str] = []

    for ids in groups.values():
        rng.shuffle(ids)
        n_take = round(len(ids) * test_fraction)
        # Clamp: single-member groups go to remainder
        if len(ids) < 2:
            n_take = 0
        n_take = min(n_take, len(ids) - 1)   # keep at least 1 for train/val
        test_ids.extend(ids[:n_take])
        remainder.extend(ids[n_take:])

    # Trim / pad to exactly n_test (rounding may drift by a few)
    rng.shuffle(test_ids)
    shortage = n_test - len(test_ids)
    if shortage > 0:
        # Pull extra from remainder (shuffle remainder first)
        rng.shuffle(remainder)
        extra = remainder[:shortage]
        remainder = remainder[shortage:]
        test_ids.extend(extra)
    elif shortage < 0:
        # Put excess back in remainder
        excess = test_ids[n_test:]
        test_ids = test_ids[:n_test]
        remainder.extend(excess)

    train_ids, val_ids = multilabel_stratified_split(
        remainder, labels_df, val_split=val_split, random_state=random_state
    )

    return test_ids, train_ids, val_ids


def get_stratified_sampler(labels_df: pd.DataFrame,
                           image_ids: List[str]) -> WeightedRandomSampler:
    """Weighted sampler to balance ICH-positive/negative batches."""
    labels = labels_df.loc[image_ids, 'any'].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1)
    sample_weights = torch.DoubleTensor(class_weights[labels])
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


# ── Dataset ───────────────────────────────────────────────────────────────────

class OneChannelDataset(Dataset):
    """Dataset for 1-channel CT .npz tensors (classification only)."""

    def __init__(self, image_ids: List[str], labels_df: pd.DataFrame,
                 cache_dir: str, image_size: int = 512, augment: bool = False):
        self.image_ids = image_ids
        self.labels_df = labels_df
        self.cache_dir = cache_dir
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        npz_path = os.path.join(self.cache_dir, f"{image_id}.npz")

        zero_image = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
        zero_labels = np.zeros(len(LABEL_COLS), dtype=np.float32)

        try:
            data = np.load(npz_path)
            image_norm = data['image_norm'].astype(np.float32)   # (H, W) in [0, 1]
        except Exception as e:
            logging.warning(f"Failed to load {npz_path}: {e}")
            raise

        # Resize if needed
        h, w = image_norm.shape
        if h != self.image_size or w != self.image_size:
            zh, zw = self.image_size / h, self.image_size / w
            image_norm = ndimage.zoom(image_norm, (zh, zw), order=1)

        # ── Geometric augmentation ────────────────────────────────────────────
        if self.augment:
            # Horizontal flip
            if np.random.random() > 0.5:
                image_norm = np.flip(image_norm, axis=1).copy()

            # Random rot90
            k = np.random.randint(0, 4)
            if k > 0:
                image_norm = np.rot90(image_norm, k).copy()

        # Add channel dim: (H, W) → (1, H, W)
        image_1ch = image_norm[np.newaxis]   # (1, H, W) float32

        cls_labels = self.labels_df.loc[image_id, LABEL_COLS].values.astype(np.float32)

        return (torch.from_numpy(image_1ch),
                torch.from_numpy(cls_labels))


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir', type=str,
                        default='/home/johnb/cache_1ch')
    parser.add_argument('--labels-csv', type=str,
                        default='/home/johnb/Documents/stage_2_train.csv')
    parser.add_argument('--num-samples', type=int, default=5)
    args = parser.parse_args()

    print(f"Scanning cache: {args.cache_dir}")
    available_ids = set()
    try:
        with os.scandir(args.cache_dir) as it:
            for entry in it:
                if entry.name.endswith('.npz') and entry.is_file():
                    available_ids.add(entry.name[:-4])
    except FileNotFoundError:
        print(f"Cache directory not found: {args.cache_dir}")
        sys.exit(1)

    print(f"Found {len(available_ids)} cached files")
    if not available_ids:
        sys.exit(1)

    labels_df = load_labels(args.labels_csv, available_ids)
    image_ids = labels_df.index.tolist()[:args.num_samples]

    ds = OneChannelDataset(image_ids, labels_df, args.cache_dir, augment=True)

    for i in range(len(ds)):
        img, cls = ds[i]
        print(f"  [{i}] image: {tuple(img.shape)} {img.dtype}  "
              f"cls: {tuple(cls.shape)} {cls.dtype}  "
              f"img range: [{img.min():.3f}, {img.max():.3f}]")
        assert img.shape == (1, 512, 512), f"Expected (1,512,512), got {img.shape}"
        assert cls.shape[0] == 6, f"Expected 6 labels, got {cls.shape}"

    print("Dataset test PASSED")
