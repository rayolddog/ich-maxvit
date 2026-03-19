#!/usr/bin/env python3
"""
Evaluate the MaxViT ICH classifier on the held-out test set.

Computes per-class metrics at the Youden-optimal threshold:
  AUC, prevalence, sensitivity, specificity, PPV, NPV, LR+, LR-

Results are saved to checkpoints_maxvit/test_metrics.json and used by
ich_dicom_sr.py instead of hardcoded performance constants.

Usage:
    python evaluate_maxvit_test.py
    python evaluate_maxvit_test.py --checkpoint checkpoints_maxvit/best_maxvit_ich.pth
    python evaluate_maxvit_test.py --output-json checkpoints_maxvit/test_metrics.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from dataclasses import dataclass, asdict
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_maxvit import MaxVITDataset
from dataset_1ch import load_labels, LABEL_COLS

KAGGLE_LABEL_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 2.0])

CLASS_DISPLAY = {
    "epidural":         "Epidural hematoma",
    "intraparenchymal": "Intraparenchymal hemorrhage",
    "intraventricular": "Intraventricular hemorrhage",
    "subarachnoid":     "Subarachnoid hemorrhage",
    "subdural":         "Subdural hematoma",
    "any":              "Any intracranial hemorrhage",
}


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class ClassMetrics:
    auc:         float
    prevalence:  float
    sensitivity: float
    specificity: float
    ppv:         float
    npv:         float
    lr_positive: float
    lr_negative: float
    threshold:   float
    n_positive:  int
    n_total:     int


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden J statistic: maximise sensitivity + specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def compute_class_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> ClassMetrics:
    auc       = float(roc_auc_score(y_true, y_prob))
    threshold = find_optimal_threshold(y_true, y_prob)
    y_pred    = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    ppv         = tp / (tp + fp + 1e-10)
    npv         = tn / (tn + fn + 1e-10)
    fpr_val     = 1.0 - specificity
    lr_pos      = sensitivity / (fpr_val + 1e-10) if fpr_val > 1e-10 else float("inf")
    fnr         = 1.0 - sensitivity
    lr_neg      = fnr / (specificity + 1e-10)

    return ClassMetrics(
        auc         = round(auc, 6),
        prevalence  = round(float(y_true.mean()), 6),
        sensitivity = round(float(sensitivity), 6),
        specificity = round(float(specificity), 6),
        ppv         = round(float(ppv), 6),
        npv         = round(float(npv), 6),
        lr_positive = round(float(lr_pos), 4) if lr_pos < 1e6 else float("inf"),
        lr_negative = round(float(lr_neg), 6),
        threshold   = round(float(threshold), 6),
        n_positive  = int(y_true.sum()),
        n_total     = int(len(y_true)),
    )


def weighted_log_loss(y_true: np.ndarray, y_prob: np.ndarray,
                      weights: np.ndarray = KAGGLE_LABEL_WEIGHTS,
                      epsilon: float = 1e-15) -> float:
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    sample_losses = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return float(np.average(sample_losses.mean(axis=0), weights=weights))


# ── Model loading ─────────────────────────────────────────────────────────────

def load_maxvit(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    import timm
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = timm.create_model(
        "maxvit_base_tf_224.in1k",
        pretrained  = False,
        in_chans    = 1,
        num_classes = len(LABEL_COLS),
    )
    state = ckpt["model_state_dict"]
    try:
        model.load_state_dict(state)
    except RuntimeError:
        sample_key = next(iter(state))
        if sample_key.startswith("_orig_mod."):
            state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)

    model.eval()
    model.to(device)
    epoch    = ckpt.get("epoch", "?")
    best_auc = ckpt.get("best_auc", 0.0)
    print(f"  Checkpoint loaded — epoch {epoch}, val AUC {best_auc:.4f}")
    return model, ckpt


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(model, dataloader, device) -> Tuple[np.ndarray, np.ndarray]:
    amp_dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda"),
                                    dtype=amp_dtype):
                logits = model(inputs)
            probs  = torch.sigmoid(logits).cpu().float().numpy()
            labels = labels.numpy()

            valid = np.isfinite(probs).all(axis=1)
            if valid.any():
                all_probs.append(probs[valid])
                all_labels.append(labels[valid])

            if (batch_idx + 1) % 100 == 0:
                done = min((batch_idx + 1) * dataloader.batch_size, len(dataloader.dataset))
                print(f"  {done:>7,} / {len(dataloader.dataset):,} slices processed",
                      end="\r", flush=True)

    print()
    return (np.concatenate(all_labels, axis=0),
            np.concatenate(all_probs,  axis=0))


# ── Report ────────────────────────────────────────────────────────────────────

def print_results(metrics: Dict[str, ClassMetrics], kaggle_wll: float,
                  mean_auc: float):
    W = 116
    print("\n" + "=" * W)
    print("TEST SET EVALUATION — MaxViT ICH Classifier")
    print("=" * W)
    hdr = (f"  {'Class':<26} {'N+':>6} {'N':>8} {'Prev':>7} {'AUC':>7} "
           f"{'Sens':>7} {'Spec':>7} {'PPV':>7} {'NPV':>7} "
           f"{'LR+':>8} {'LR-':>8} {'Thr':>7}")
    print(hdr)
    print("  " + "-" * (W - 2))
    for col in LABEL_COLS:
        m   = metrics[col]
        lrp = f"{m.lr_positive:8.2f}" if m.lr_positive < 1e5 else "     inf"
        print(f"  {CLASS_DISPLAY.get(col, col):<26} {m.n_positive:>6,} {m.n_total:>8,} "
              f"{m.prevalence:7.4f} {m.auc:7.4f} "
              f"{m.sensitivity:7.4f} {m.specificity:7.4f} "
              f"{m.ppv:7.4f} {m.npv:7.4f} "
              f"{lrp} {m.lr_negative:8.5f} {m.threshold:7.4f}")
    print("  " + "-" * (W - 2))
    print(f"  {'MEAN AUC':<26} {'':>6} {'':>8} {'':>7} {mean_auc:7.4f}")
    print(f"\n  Kaggle Weighted Log-Loss: {kaggle_wll:.5f}")
    print("=" * W + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MaxViT ICH model on the held-out test set"
    )
    parser.add_argument("--checkpoint",
        default="./checkpoints_maxvit/best_maxvit_ich.pth")
    parser.add_argument("--splits-file",
        default="./checkpoints_1ch/data_splits.json")
    parser.add_argument("--test-dir",
        default="/home/justinolddog/cache_medhu_test")
    parser.add_argument("--labels-csv",
        default="/home/justinolddog/FullICH/RSNAICH/stage_2_train.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers",    type=int, default=16)
    parser.add_argument("--output-json",
        default="./checkpoints_maxvit/test_metrics.json")
    args = parser.parse_args()

    # ── Load splits ────────────────────────────────────────────────────────────
    print(f"Loading splits: {args.splits_file}")
    with open(args.splits_file) as f:
        splits = json.load(f)
    test_ids = splits["test_ids"]
    print(f"  Split test IDs: {len(test_ids):,}")

    # Filter to IDs present in test cache
    present = {
        e.name[:-4]
        for e in os.scandir(args.test_dir)
        if e.name.endswith(".npz") and e.is_file()
    }
    test_ids = [x for x in test_ids if x in present]
    print(f"  IDs present in cache: {len(test_ids):,}")

    # ── Labels ────────────────────────────────────────────────────────────────
    labels_df = load_labels(args.labels_csv, set(test_ids))
    test_ids  = [x for x in test_ids if x in labels_df.index]
    print(f"  IDs with labels: {len(test_ids):,}")

    # ── Dataset & loader ──────────────────────────────────────────────────────
    test_ds = MaxVITDataset(test_ids, labels_df, args.test_dir, augment=False)
    loader  = DataLoader(
        test_ds,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.workers,
        pin_memory  = True,
    )

    # ── Device & model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading checkpoint: {args.checkpoint}")
    model, ckpt = load_maxvit(args.checkpoint, device)

    # ── Inference ─────────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(test_ids):,} slices...")
    t0 = time.perf_counter()
    all_labels, all_probs = run_inference(model, loader, device)
    inference_seconds = time.perf_counter() - t0
    throughput = len(test_ids) / inference_seconds
    print(f"  Done. Labels shape: {all_labels.shape}, Probs shape: {all_probs.shape}")
    print(f"  Inference time : {inference_seconds/60:.1f} min "
          f"({inference_seconds:.1f} s,  {throughput:.0f} slices/s)")

    # ── Per-class metrics ─────────────────────────────────────────────────────
    metrics: Dict[str, ClassMetrics] = {}
    for i, col in enumerate(LABEL_COLS):
        try:
            metrics[col] = compute_class_metrics(all_labels[:, i], all_probs[:, i])
        except Exception as exc:
            print(f"  Warning: could not compute metrics for {col}: {exc}")

    kaggle_wll = weighted_log_loss(all_labels, all_probs)
    mean_auc   = float(np.mean([m.auc for m in metrics.values()]))

    # ── Print ─────────────────────────────────────────────────────────────────
    print_results(metrics, kaggle_wll, mean_auc)

    # ── Save JSON ─────────────────────────────────────────────────────────────
    output = {
        "model":             "MaxViT-Base (timm maxvit_base_tf_224.in1k), 118.7M parameters",
        "training_data":     "RSNA ICH dataset — 544,685 training images, 6-class multi-label",
        "checkpoint":        args.checkpoint,
        "epoch":             ckpt.get("epoch", "?"),
        "val_auc":           ckpt.get("best_auc", None),
        "n_test":            len(test_ids),
        "mean_auc":          round(mean_auc, 6),
        "kaggle_wll":        round(kaggle_wll, 6),
        "inference_seconds": round(inference_seconds, 1),
        "slices_per_second": round(throughput, 1),
        "per_class":         {col: asdict(m) for col, m in metrics.items()},
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Metrics saved → {args.output_json}")


if __name__ == "__main__":
    main()
