#!/usr/bin/env python3
"""
ich_metrics.py — the single source of clinical metrics for the pipeline.

Loads the canonical test-set metrics (checkpoints_maxvit/test_metrics.json — the
same artifact the README table and inference thresholds use) and recomputes PPV
and NPV at any assumed local prevalence via Bayes. This replaces the metrics
computation that used to live inside the old ich_dicom_sr.py, so there is exactly
one place clinical numbers come from.

    metrics_for("subdural", prevalence=0.06) -> {sensitivity, specificity,
        ppv, npv, auc, threshold, prevalence, class}

PPV/NPV are prevalence-dependent (the Bayesian trust-calibration point); AUC,
sensitivity, specificity, and the operating threshold are intrinsic to the model
and read straight from the artifact.
"""
import json
import os

METRICS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "checkpoints_maxvit", "test_metrics.json")


def _per_class(path=METRICS_PATH):
    with open(path) as f:
        return json.load(f)["per_class"]


def metrics_for(class_key, prevalence, path=METRICS_PATH):
    """Metrics for one class at an assumed prevalence. class_key falls back to
    'any' if unknown. PPV/NPV recomputed at `prevalence` from the model's
    intrinsic sensitivity/specificity."""
    per = _per_class(path)
    v = per.get(class_key, per["any"])
    sens, spec = float(v["sensitivity"]), float(v["specificity"])
    p = max(0.0, min(1.0, float(prevalence)))
    tp, fp = sens * p, (1 - spec) * (1 - p)
    tn, fn = spec * (1 - p), (1 - sens) * p
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    return {
        "class":       class_key,
        "sensitivity": sens,
        "specificity": spec,
        "ppv":         ppv,
        "npv":         npv,
        "auc":         float(v["auc"]),
        "threshold":   float(v["threshold"]),
        "prevalence":  p,
        "lr_positive": float(v.get("lr_positive", 0.0)),
        "lr_negative": float(v.get("lr_negative", 0.0)),
    }


def dominant_positive_class(study_level):
    """Highest-probability positive subtype key (excluding 'any'); 'any' if none."""
    pos = [(c, v["prob"]) for c, v in study_level.items()
           if c != "any" and v.get("positive")]
    return max(pos, key=lambda x: x[1])[0] if pos else "any"


if __name__ == "__main__":
    import sys
    cls = sys.argv[1] if len(sys.argv) > 1 else "subdural"
    prev = float(sys.argv[2]) if len(sys.argv) > 2 else 0.06
    m = metrics_for(cls, prev)
    print(f"{cls} at {prev*100:.0f}% prevalence: "
          f"Sens {m['sensitivity']:.3f}, Spec {m['specificity']:.3f}, "
          f"PPV {m['ppv']*100:.1f}%, NPV {m['npv']*100:.2f}%")
