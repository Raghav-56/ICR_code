from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)



def compute_binary_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }



def predict_with_threshold(model: Any, x, threshold: float = 0.5):
    y_prob = model.predict_proba(x)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_pred, y_prob


def select_threshold_by_f1(y_true, y_prob) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5

    # precision/recall has one extra element; align to threshold positions.
    p = precision[:-1]
    r = recall[:-1]
    f1 = (2 * p * r) / np.clip(p + r, 1e-12, None)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])



def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    arr = np.asarray(values)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        boots.append(float(np.mean(arr[idx])))
    return {
        "mean": float(np.mean(arr)),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
    }
