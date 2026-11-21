"""Evaluation metrics and inference benchmarking utility for MLForge.

Provides standard binary classification metrics, threshold-independent metrics (ROC-AUC, PR-AUC),
validation-only classification threshold tuning, and multi-run latency benchmarking.
"""

import time
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_threshold_independent_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Calculate ROC-AUC and Average Precision (PR-AUC), which are threshold-independent.

    Args:
        y_true: Binary ground truth array (0 or 1).
        y_prob: Predicted positive class probabilities in [0.0, 1.0].

    Returns:
        Dictionary containing 'roc_auc' and 'pr_auc'.
    """
    roc_auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Calculate threshold-dependent classification metrics.

    Args:
        y_true: Binary ground truth array (0 or 1).
        y_prob: Predicted positive class probabilities in [0.0, 1.0].
        threshold: Decision threshold for positive prediction (default 0.50).

    Returns:
        Dictionary of accuracy, precision, recall, f1, and confusion matrix.
    """
    y_pred = (y_prob >= threshold).astype(int)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "threshold": round(threshold, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
        },
    }


def find_optimal_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_threshold: float = 0.05,
    max_threshold: float = 0.95,
    num_steps: int = 181,
) -> Tuple[float, float]:
    """Find the classification threshold that maximizes F1 score on validation data.

    Args:
        y_true: Validation ground truth array.
        y_prob: Validation predicted probabilities.
        min_threshold: Lower bound for threshold sweep.
        max_threshold: Upper bound for threshold sweep.
        num_steps: Number of candidate thresholds evaluated.

    Returns:
        Tuple of (best_threshold, best_f1).
    """
    thresholds = np.linspace(min_threshold, max_threshold, num_steps)
    best_threshold = 0.50
    best_f1 = -1.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        score = float(f1_score(y_true, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)

    return round(best_threshold, 4), round(best_f1, 4)


def benchmark_inference_latency(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_sample: np.ndarray,
    num_warmups: int = 3,
    num_repeats: int = 10,
) -> Dict[str, float]:
    """Measure inference latency using warm-ups and repeated measurements.

    Args:
        predict_fn: Callable taking input matrix and returning predictions.
        X_sample: Evaluation batch (e.g. full validation set).
        num_warmups: Warm-up iterations to prime caches and JIT.
        num_repeats: Measurement iterations to compute stable statistics.

    Returns:
        Dictionary with median_batch_latency_ms and per_sample_latency_ms.
    """
    batch_size = len(X_sample)

    # Warm-up runs
    for _ in range(num_warmups):
        _ = predict_fn(X_sample)

    # Measured runs
    latencies: List[float] = []
    for _ in range(num_repeats):
        start = time.perf_counter()
        _ = predict_fn(X_sample)
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    median_latency_sec = float(np.median(latencies))
    median_batch_ms = median_latency_sec * 1000.0
    per_sample_ms = (median_latency_sec / batch_size) * 1000.0
    per_sample_us = per_sample_ms * 1000.0

    return {
        "num_repeats": num_repeats,
        "batch_size": batch_size,
        "median_batch_latency_ms": round(median_batch_ms, 3),
        "per_sample_latency_ms": round(per_sample_ms, 5),
        "per_sample_latency_us": round(per_sample_us, 2),
    }
