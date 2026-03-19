"""
MaxViT ICH classification — medium-window HU tensors.

Architecture : MaxViT (timm) — in_chans=1, 224×224 input
Precision    : BF16 (Blackwell-native; wider dynamic range than FP16)
Compile      : torch.compile() for ~20–30% throughput gain on GB10
Input        : medium-window .npz cache (ICH_MedHU / cache_medhu_train)
Labels       : RSNA stage_2_train.csv, 6 multi-label classes
Splits       : reuses existing data_splits.json from 1ch pipeline

Transfer the following files to the DGX Spark alongside this script:
  dataset_maxvit.py
  dataset_1ch.py         (provides load_labels, LABEL_COLS, split utilities)
  stage_2_train.csv      (or point --labels-csv at your copy)
  data_splits.json       (or point --splits-file at your copy)
  cache_medhu_train/     (training + val .npz tensors)
  cache_medhu_test/      (held-out test .npz tensors)

Install:
  pip install torch torchvision timm scikit-learn pandas scipy

Usage:
  python train_maxvit.py \\
      --cache-dir    /data/cache_medhu_train \\
      --test-dir     /data/cache_medhu_test \\
      --labels-csv   /data/stage_2_train.csv \\
      --splits-file  /data/data_splits.json \\
      --checkpoint-dir ./checkpoints_maxvit \\
      --batch-size 256 --workers 16 --epochs 50

Resume:
  python train_maxvit.py ... --resume ./checkpoints_maxvit/latest_maxvit_ich.pth
"""

import os
import sys
import json
import time
import argparse
import warnings
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

warnings.filterwarnings('ignore')

# ── Local imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_maxvit import MaxVITDataset
from dataset_1ch import (
    LABEL_COLS, load_labels, split_three_way, get_stratified_sampler
)

# ── Constants ──────────────────────────────────────────────────────────────────
KAGGLE_LABEL_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])
MODEL_TAG            = 'maxvit_ich'


# ── Likelihood metrics ─────────────────────────────────────────────────────────

@dataclass
class LikelihoodMetrics:
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    lr_positive: float
    lr_negative: float
    prevalence: float
    threshold: float


def find_optimal_threshold(y_true: np.ndarray,
                           y_prob: np.ndarray) -> Tuple[float, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], j_scores[best_idx]


def calculate_likelihood_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                                 threshold: Optional[float] = None
                                 ) -> LikelihoodMetrics:
    if threshold is None:
        threshold, _ = find_optimal_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    ppv = tp / (tp + fp + 1e-10)
    npv = tn / (tn + fn + 1e-10)
    fpr_val = 1 - specificity
    lr_pos  = sensitivity / (fpr_val + 1e-10) if fpr_val > 0 else float('inf')
    lr_neg  = (1 - sensitivity) / (specificity + 1e-10)

    return LikelihoodMetrics(
        sensitivity=sensitivity, specificity=specificity,
        ppv=ppv, npv=npv,
        lr_positive=lr_pos, lr_negative=lr_neg,
        prevalence=float(y_true.sum() / len(y_true)),
        threshold=threshold,
    )


def weighted_log_loss(y_true: np.ndarray, y_prob: np.ndarray,
                      weights: np.ndarray = KAGGLE_LABEL_WEIGHTS,
                      epsilon: float = 1e-15) -> float:
    valid = np.isfinite(y_prob).all(axis=1)
    y_true, y_prob = y_true[valid], y_prob[valid]
    if len(y_true) == 0:
        return float('nan')
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    sample_losses = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return float(np.average(sample_losses.mean(axis=0), weights=weights))


# ── Loss ───────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss for multi-label binary classification."""

    def __init__(self, alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2.0, label_smoothing: float = 0.0,
                 reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing > 0:
            targets = (targets * (1 - self.label_smoothing)
                       + 0.5 * self.label_smoothing)

        bce = F.binary_cross_entropy_with_logits(logits, targets,
                                                  reduction='none')
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = (self.alpha * targets
                       + (1 - self.alpha) * (1 - targets))
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(model_name: str, num_classes: int, dropout: float,
                pretrained: bool) -> nn.Module:
    """Build MaxViT model with 1-channel input.

    timm adapts pretrained 3-channel conv1 weights to 1-channel by
    summing (or averaging) across the channel axis — preserving ImageNet
    spatial features rather than training from zero.

    Common model_name choices (all 224×224 native):
      maxvit_tiny_tf_224.in1k   ~31M params  (fastest)
      maxvit_small_tf_224.in1k  ~68M params
      maxvit_base_tf_224.in1k   ~120M params (recommended default)
      maxvit_large_tf_224.in1k  ~212M params
    """
    import timm

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=1,          # timm adapts conv1 weights automatically
        num_classes=num_classes,
        drop_rate=dropout,
    )

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model       : {model_name}")
    print(f"  Parameters  : {num_params:,}")
    print(f"  Pretrained  : {pretrained}")
    return model


# ── CutMix augmentation ────────────────────────────────────────────────────────

def cutmix_batch(inputs: torch.Tensor, targets: torch.Tensor,
                 alpha: float = 1.0
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """CutMix: paste a random box from one image into another.

    Multi-label targets are mixed proportionally to the box area ratio.
    CutMix is particularly effective for ICH because hemorrhage is often
    a small, spatially localised region — it forces the model to rely on
    the actual density pattern rather than global context shortcuts.
    """
    lam = np.random.beta(alpha, alpha)
    B = inputs.size(0)
    rand_idx = torch.randperm(B, device=inputs.device)

    _, _, H, W = inputs.shape
    cut_h = int(H * np.sqrt(1 - lam))
    cut_w = int(W * np.sqrt(1 - lam))
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, H)

    inputs[:, :, y1:y2, x1:x2] = inputs[rand_idx, :, y1:y2, x1:x2]

    # Actual mix ratio (box may be clipped at image edges)
    lam_actual = 1 - (x2 - x1) * (y2 - y1) / (H * W)
    targets = lam_actual * targets + (1 - lam_actual) * targets[rand_idx]
    return inputs, targets


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, criterion, optimizer, device,
                scaler=None, use_cutmix: bool = True,
                cutmix_prob: float = 0.5) -> Tuple[float, int]:
    model.train()
    loss_sum   = 0.0
    n_batches  = 0
    n_tensors  = 0
    last_mile  = 0

    optimizer.zero_grad()
    amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

    for batch_idx, (inputs, cls_labels) in enumerate(dataloader):
        inputs     = inputs.to(device, non_blocking=True)
        cls_labels = cls_labels.to(device, non_blocking=True)

        # CutMix — apply with probability cutmix_prob
        if use_cutmix and np.random.random() < cutmix_prob:
            inputs, cls_labels = cutmix_batch(inputs, cls_labels)

        if scaler is not None:
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                logits = model(inputs)
                loss   = criterion(logits, cls_labels)

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            logits = model(inputs)
            loss   = criterion(logits, cls_labels)

            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        loss_sum  += loss.item()
        n_batches += 1
        n_tensors += inputs.size(0)

        # Progress milestone every 10,000 tensors
        mile = n_tensors // 10000
        if mile > last_mile:
            last_mile = mile
            avg_loss  = loss_sum / n_batches
            ep_label  = getattr(train_epoch, '_current_epoch', '?')
            print(f"  [ep {ep_label}] {n_tensors:,}  loss {avg_loss:.4f}",
                  flush=True)

    avg_loss = loss_sum / n_batches if n_batches > 0 else float('nan')
    return avg_loss, n_tensors


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_batches    = 0
    all_labels   = []
    all_probs    = []

    amp_dtype = torch.bfloat16 if device.type == 'cuda' else torch.float32

    with torch.no_grad():
        for inputs, cls_labels in dataloader:
            inputs     = inputs.to(device, non_blocking=True)
            cls_labels = cls_labels.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda'),
                                    dtype=amp_dtype):
                logits = model(inputs)
                loss   = criterion(logits, cls_labels)

            if not torch.isfinite(loss):
                continue

            probs   = torch.sigmoid(logits)
            running_loss += loss.item()
            n_batches    += 1

            probs_np  = probs.cpu().float().numpy()
            labels_np = cls_labels.cpu().float().numpy()
            valid = np.isfinite(probs_np).all(axis=1)
            if valid.any():
                all_labels.append(labels_np[valid])
                all_probs.append(probs_np[valid])

    avg_loss = running_loss / n_batches if n_batches > 0 else float('nan')
    if not all_labels:
        return avg_loss, {}, 0.0, {}, float('nan')

    all_labels = np.concatenate(all_labels, axis=0)
    all_probs  = np.concatenate(all_probs,  axis=0)

    aucs = {}
    lm   = {}
    for i, col in enumerate(LABEL_COLS):
        try:
            aucs[col] = roc_auc_score(all_labels[:, i], all_probs[:, i])
            lm[col]   = calculate_likelihood_metrics(
                all_labels[:, i], all_probs[:, i])
        except ValueError:
            aucs[col] = 0.5
            lm[col]   = LikelihoodMetrics(0, 0, 0, 0, 1, 1, 0, 0.5)

    valid_aucs = [v for v in aucs.values() if not np.isnan(v)]
    mean_auc   = float(np.mean(valid_aucs)) if valid_aucs else 0.0
    wll        = weighted_log_loss(all_labels, all_probs)

    return avg_loss, aucs, mean_auc, lm, wll


# ── Report ─────────────────────────────────────────────────────────────────────

def print_epoch_table(epoch: int, aucs: Dict, lm: Dict, wll: float):
    print(f"\nEpoch {epoch:02d} — val")
    header = (f"  {'Class':<22} {'AUC':>6} {'Prev':>6} "
              f"{'PPV':>6} {'NPV':>6} {'LR+':>6} {'LR-':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    auc_vals = []
    for col in LABEL_COLS:
        auc = aucs.get(col, float('nan'))
        m   = lm.get(col)
        if m is not None:
            lr_pos = (f"{m.lr_positive:6.1f}"
                      if m.lr_positive != float('inf') else "  inf")
            print(f"  {col:<22} {auc:6.3f} {m.prevalence:6.3f} "
                  f"{m.ppv:6.3f} {m.npv:6.3f} {lr_pos} {m.lr_negative:6.3f}")
        else:
            print(f"  {col:<22} {auc:6.3f}")
        if not np.isnan(auc):
            auc_vals.append(auc)

    mean_auc = float(np.mean(auc_vals)) if auc_vals else float('nan')
    print(f"  {'MEAN':<22} {mean_auc:6.3f}")
    print(f"  Kaggle WLL : {wll:.5f}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='MaxViT ICH classification — medium-window HU')

    # ── Paths ──────────────────────────────────────────────────────────────────
    parser.add_argument('--cache-dir',       default='/data/cache_medhu_train',
                        help='Train+val .npz cache dir (ICH_MedHU/cache_medhu_train)')
    parser.add_argument('--test-dir',        default='/data/cache_medhu_test',
                        help='Test .npz cache dir (ICH_MedHU/cache_medhu_test)')
    parser.add_argument('--labels-csv',      default='/data/stage_2_train.csv')
    parser.add_argument('--splits-file',     default='/data/data_splits.json',
                        help='Existing data_splits.json from 1ch pipeline')
    parser.add_argument('--checkpoint-dir',  default='./checkpoints_maxvit')
    parser.add_argument('--resume',          default=None,
                        help='Path to checkpoint .pth to resume from')

    # ── Model ──────────────────────────────────────────────────────────────────
    parser.add_argument('--model-name',  default='maxvit_base_tf_224.in1k',
                        help='timm MaxViT model name')
    parser.add_argument('--pretrained',  action='store_true',  default=True)
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.add_argument('--dropout',     type=float, default=0.3)
    parser.add_argument('--compile',     action='store_true', default=True,
                        help='torch.compile() for ~20–30%% throughput gain')
    parser.add_argument('--no-compile',  dest='compile', action='store_false')

    # ── Training ───────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',         type=int,   default=50)
    parser.add_argument('--batch-size',     type=int,   default=256,
                        help='Large default for DGX Spark 128GB; reduce if OOM')
    parser.add_argument('--workers',        type=int,   default=16,
                        help='DataLoader workers; match CPU core count')
    parser.add_argument('--lr',             type=float, default=2e-4)
    parser.add_argument('--weight-decay',   type=float, default=0.05)
    parser.add_argument('--warmup-epochs',  type=int,   default=5)
    parser.add_argument('--focal-gamma',    type=float, default=2.0)
    parser.add_argument('--label-smoothing',type=float, default=0.05)
    parser.add_argument('--cutmix',         action='store_true', default=True)
    parser.add_argument('--no-cutmix',      dest='cutmix', action='store_false')
    parser.add_argument('--cutmix-prob',    type=float, default=0.5)
    parser.add_argument('--stratified-sampling', action='store_true')
    parser.add_argument('--num-images',     type=int,   default=0,
                        help='Limit dataset (0 = all); useful for quick tests')

    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ── Device setup ───────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice      : {device}")
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"GPU         : {props.name}  ({props.total_memory // 1024**3} GB)")
        # BF16 is preferred on Blackwell (GB10) and Hopper (H100) architectures
        print(f"AMP dtype   : bfloat16")
    else:
        print("WARNING: CUDA not available — training will be slow")

    # ── Scan cache ─────────────────────────────────────────────────────────────
    print(f"\nScanning train/val cache: {args.cache_dir}")
    trainval_available = set()
    with os.scandir(args.cache_dir) as it:
        for entry in it:
            if entry.name.endswith('.npz') and entry.is_file():
                trainval_available.add(entry.name[:-4])
    print(f"  Found {len(trainval_available):,} train/val .npz files")

    if not trainval_available:
        print("ERROR: no cached files found. Check --cache-dir.")
        return

    # ── Labels ─────────────────────────────────────────────────────────────────
    labels_df = load_labels(args.labels_csv, trainval_available)

    if args.num_images > 0 and args.num_images < len(labels_df):
        labels_df = labels_df.head(args.num_images)
        print(f"\nUsing {len(labels_df):,} images (limited mode)")

    # ── Splits ─────────────────────────────────────────────────────────────────
    image_ids = labels_df.index.tolist()

    if os.path.exists(args.splits_file):
        print(f"\nLoading splits from {args.splits_file}")
        with open(args.splits_file) as f:
            splits = json.load(f)
        available = set(image_ids)
        train_ids = [x for x in splits['train_ids'] if x in available]
        val_ids   = [x for x in splits['val_ids']   if x in available]
        print(f"  train: {len(train_ids):,}  val: {len(val_ids):,}")
    else:
        print(f"\nWARNING: splits file not found at {args.splits_file}")
        print("Generating new 85/15 stratified split (random_state=42)...")
        # Use all train/val; create a dummy test pool the same way as 1ch pipeline
        _, train_ids, val_ids = split_three_way(
            image_ids, labels_df, n_test=200000, val_split=0.15, random_state=42)
        splits_out = os.path.join(args.checkpoint_dir, 'data_splits_maxvit.json')
        with open(splits_out, 'w') as f:
            json.dump({'train_ids': train_ids, 'val_ids': val_ids}, f)
        print(f"  Saved new splits → {splits_out}")
        print(f"  train: {len(train_ids):,}  val: {len(val_ids):,}")

    # ── Datasets & loaders ─────────────────────────────────────────────────────
    train_ds = MaxVITDataset(train_ids, labels_df, args.cache_dir, augment=True)
    val_ds   = MaxVITDataset(val_ids,   labels_df, args.cache_dir, augment=False)

    loader_kw = dict(
        batch_size      = args.batch_size,
        num_workers     = args.workers,
        pin_memory      = True,
        persistent_workers = True,  # keep workers alive across epochs
        prefetch_factor = 4,        # prefetch ahead of GPU consumption
    )

    if args.stratified_sampling:
        sampler      = get_stratified_sampler(labels_df, train_ids)
        train_loader = DataLoader(train_ds, sampler=sampler, **loader_kw)
    else:
        train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)

    val_loader = DataLoader(val_ds, shuffle=False, **loader_kw)

    # ── Model ──────────────────────────────────────────────────────────────────
    print(f"\nBuilding model...")
    model = build_model(args.model_name, num_classes=len(LABEL_COLS),
                        dropout=args.dropout, pretrained=args.pretrained)
    model = model.to(device)

    # torch.compile: fuses ops into efficient kernels (CUDA graphs on Blackwell)
    # 'reduce-overhead' mode minimises Python overhead for fast dataloaders
    if args.compile and device.type == 'cuda':
        print("  Compiling model (torch.compile)...")
        model = torch.compile(model, mode='reduce-overhead')
        print("  Compile done.")

    # ── Loss ───────────────────────────────────────────────────────────────────
    pos_counts = labels_df[LABEL_COLS].sum()
    neg_counts = len(labels_df) - pos_counts
    alpha = torch.tensor(
        (neg_counts / (pos_counts + neg_counts)).values, dtype=torch.float32
    ).to(device)
    criterion = FocalLoss(alpha=alpha, gamma=args.focal_gamma,
                          label_smoothing=args.label_smoothing)

    # ── Optimizer & scheduler ──────────────────────────────────────────────────
    # AdamW with decoupled weight decay; higher wd than SE-ResNeXt because
    # MaxViT attention parameters are more prone to overfitting
    optimizer = optim.AdamW(model.parameters(),
                            lr=args.lr, weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        """Linear warmup then cosine decay."""
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = ((epoch - args.warmup_epochs)
                    / max(args.epochs - args.warmup_epochs, 1))
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # GradScaler: use BF16 via autocast; scaler mostly a no-op for BF16
    # but kept for compatibility if FP16 is selected manually
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    # ── Resume ─────────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_auc     = 0.0
    prev_history = None

    if args.resume and os.path.isfile(args.resume):
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        # compiled model state dict keys start with '_orig_mod.' — handle both
        state = ckpt['model_state_dict']
        try:
            model.load_state_dict(state)
        except RuntimeError:
            # Checkpoint saved with bare keys (from _orig_mod.state_dict()) but
            # loading into a compiled model — add the prefix.
            # Also handles the reverse: strip prefix for non-compiled model.
            sample_key = next(iter(state))
            model_is_compiled = hasattr(model, '_orig_mod')
            ckpt_has_prefix = sample_key.startswith('_orig_mod.')
            if model_is_compiled and not ckpt_has_prefix:
                state = {'_orig_mod.' + k: v for k, v in state.items()}
            elif not model_is_compiled and ckpt_has_prefix:
                state = {k.replace('_orig_mod.', '', 1): v for k, v in state.items()}
            model.load_state_dict(state)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch  = ckpt['epoch'] + 1
        best_auc     = ckpt.get('best_auc', 0.0)
        prev_history = ckpt.get('history')
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if scaler and 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        print(f"  Resumed epoch {start_epoch}, best AUC {best_auc:.4f}")

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\n{'='*64}")
    print(f"MaxViT ICH classification — medium window (−200 to 200 HU)")
    print(f"  Model      : {args.model_name}")
    print(f"  Cache      : {args.cache_dir}")
    print(f"  Batch      : {args.batch_size}  LR: {args.lr}  "
          f"WD: {args.weight_decay}")
    print(f"  CutMix     : {args.cutmix} (p={args.cutmix_prob})")
    print(f"  Compile    : {args.compile}")
    print(f"{'='*64}\n")

    history = defaultdict(list)
    if prev_history:
        for k, v in prev_history.items():
            history[k] = v

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        train_epoch._current_epoch = epoch + 1

        train_loss, n_tensors = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            use_cutmix=args.cutmix, cutmix_prob=args.cutmix_prob,
        )
        val_loss, val_aucs, mean_auc, lm, val_wll = evaluate(
            model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['mean_auc'].append(mean_auc)
        history['kaggle_wll'].append(val_wll)
        for col in LABEL_COLS:
            history[f'auc_{col}'].append(val_aucs.get(col, 0.0))

        print(f"Epoch {epoch+1}/{args.epochs}  ({elapsed:.1f}s)  "
              f"LR: {current_lr:.2e}  train_loss: {train_loss:.4f}")
        print_epoch_table(epoch + 1, val_aucs, lm, val_wll)

        # ── Build checkpoint dict ──────────────────────────────────────────────
        # Get state dict: compiled models use _orig_mod prefix
        raw_state = (model._orig_mod.state_dict()
                     if hasattr(model, '_orig_mod')
                     else model.state_dict())
        ckpt_data = {
            'epoch'            : epoch,
            'model_state_dict' : raw_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_auc'         : best_auc,
            'kaggle_wll'       : val_wll,
            'val_aucs'         : val_aucs,
            'likelihood_metrics': {c: asdict(m) for c, m in lm.items()},
            'args'             : vars(args),
            'history'          : dict(history),
        }
        if scaler:
            ckpt_data['scaler_state_dict'] = scaler.state_dict()

        # ── Save best ─────────────────────────────────────────────────────────
        if mean_auc > best_auc:
            best_auc = mean_auc
            ckpt_data['best_auc'] = best_auc
            best_path = os.path.join(args.checkpoint_dir,
                                     f'best_{MODEL_TAG}.pth')
            torch.save(ckpt_data, best_path)
            print(f"  ★ New best → {best_path}  (AUC {best_auc:.4f})")

        # ── Save latest ───────────────────────────────────────────────────────
        latest_path = os.path.join(args.checkpoint_dir,
                                   f'latest_{MODEL_TAG}.pth')
        torch.save(ckpt_data, latest_path)
        print(f"  Saved latest (epoch {epoch+1})\n")

    # ── Save training history ──────────────────────────────────────────────────
    hist_path = os.path.join(args.checkpoint_dir,
                             f'{MODEL_TAG}_history.json')
    with open(hist_path, 'w') as f:
        json.dump(dict(history), f, indent=2)

    print(f"{'='*64}")
    print(f"Training complete!  Best mean AUC: {best_auc:.4f}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"{'='*64}")


if __name__ == '__main__':
    main()
