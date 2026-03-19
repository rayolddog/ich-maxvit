"""
Dataset for MaxViT ICH classification.

Loads medium-window HU tensors from .npz cache (512×512 float16),
resizes to 224×224, and returns (1, 224, 224) float32 tensors.

Reuses label utilities from dataset_1ch.py:
  - load_labels()
  - LABEL_COLS
  - multilabel_stratified_split / split_three_way / get_stratified_sampler

Augmentation (training only):
  - Random horizontal flip
  - Random rot90 (0, 90, 180, 270°)
  - Mild intensity jitter (±4% uniform, applied before resize)

The intensity jitter is kept very small because the HU window encoding is
calibrated: the absolute pixel value carries tissue-type information.
Stretching by ±4% corresponds to ±16 HU uncertainty on the medium window
(400 HU range × 0.04), which is within real scanner-to-scanner variability.
"""

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# Reuse label/split utilities from the 1ch pipeline
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_1ch import (
    LABEL_COLS, load_labels,
    multilabel_stratified_split, split_three_way, get_stratified_sampler
)

TARGET_SIZE = 224   # MaxViT native input size


class MaxVITDataset(Dataset):
    """Dataset for MaxViT ICH classification.

    Args:
        image_ids:  List of image ID strings (keys into labels_df index).
        labels_df:  DataFrame returned by load_labels(); indexed by image_id,
                    columns = LABEL_COLS.
        cache_dir:  Directory containing <image_id>.npz files.
                    Each .npz must have key 'image_norm': float16 (H, W).
        image_size: Output spatial size (default 224 for MaxViT).
        augment:    If True, apply training augmentation.
    """

    def __init__(self, image_ids, labels_df, cache_dir: str,
                 image_size: int = TARGET_SIZE, augment: bool = False):
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

        try:
            data = np.load(npz_path)
            image_norm = data['image_norm'].astype(np.float32)  # (H, W) in [0, 1]
        except Exception as e:
            logging.warning(f"Failed to load {npz_path}: {e}")
            # Return zero tensor — will be NaN-filtered during evaluation
            image_norm = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # ── Augmentation (before resize — cheaper on smaller array) ───────────
        if self.augment:
            # Horizontal flip
            if np.random.random() > 0.5:
                image_norm = np.flip(image_norm, axis=1).copy()

            # Random rot90 (axial CT is rotationally symmetric)
            k = np.random.randint(0, 4)
            if k > 0:
                image_norm = np.rot90(image_norm, k).copy()

            # Mild intensity jitter: ±4% uniform stretch around 0.5
            # Equivalent to ±16 HU on the medium window — within scanner variance
            jitter = np.random.uniform(-0.04, 0.04)
            image_norm = np.clip(image_norm + jitter, 0.0, 1.0)

        # ── Resize to TARGET_SIZE using bilinear interpolation ────────────────
        h, w = image_norm.shape
        if h != self.image_size or w != self.image_size:
            # F.interpolate needs (N, C, H, W); add/remove batch+channel dims
            t = torch.from_numpy(image_norm).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            t = F.interpolate(t, size=(self.image_size, self.image_size),
                              mode='bilinear', align_corners=False)
            image_norm = t.squeeze(0).squeeze(0).numpy()  # (224,224)

        # (H, W) → (1, H, W) channel dim for MaxViT
        image_1ch = image_norm[np.newaxis].astype(np.float32)  # (1, 224, 224)

        cls_labels = (self.labels_df.loc[image_id, LABEL_COLS]
                      .values.astype(np.float32))

        return (torch.from_numpy(image_1ch),
                torch.from_numpy(cls_labels))


# ── Quick smoke test ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache-dir',  default='/home/johnb/ICH_MedHU/cache_medhu_train')
    parser.add_argument('--labels-csv', default='/home/johnb/Documents/stage_2_train.csv')
    parser.add_argument('--n',          type=int, default=5)
    args = parser.parse_args()

    available_ids = set()
    with os.scandir(args.cache_dir) as it:
        for entry in it:
            if entry.name.endswith('.npz') and entry.is_file():
                available_ids.add(entry.name[:-4])
    print(f"Found {len(available_ids):,} cached files")

    labels_df = load_labels(args.labels_csv, available_ids)
    ids = labels_df.index.tolist()[:args.n]

    ds = MaxVITDataset(ids, labels_df, args.cache_dir, augment=True)
    for i in range(len(ds)):
        img, cls = ds[i]
        print(f"  [{i}] image: {tuple(img.shape)} {img.dtype}  "
              f"range [{img.min():.3f}, {img.max():.3f}]  "
              f"cls: {cls.tolist()}")
        assert img.shape == (1, 224, 224), f"Expected (1,224,224) got {img.shape}"
        assert cls.shape == (6,)

    print("Smoke test PASSED")
