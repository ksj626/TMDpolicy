from __future__ import annotations

import math
from typing import Any

import numpy as np


def _curve(labels: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or not len(labels) or not set(labels).issubset({0, 1}):
        raise ValueError("binary labels and scores must be nonempty and shape matched")
    positives = int(labels.sum())
    if positives == 0:
        return np.asarray([0.0]), np.asarray([math.nan])
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(1 - sorted_labels)
    recall = true_positive / positives
    precision = true_positive / (true_positive + false_positive)
    return np.concatenate(([0.0], recall)), np.concatenate(([1.0], precision))


def average_precision(labels: Any, scores: Any) -> float:
    """Step-integrated average precision, not trapezoidal PR-AUC."""

    recall, precision = _curve(labels, scores)
    return float(np.sum(np.diff(recall) * precision[1:]))


def precision_recall_auc(labels: Any, scores: Any) -> float:
    """Trapezoidal area under the empirical precision-recall curve."""

    recall, precision = _curve(labels, scores)
    return float(np.trapezoid(precision, recall))


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials < 1:
        raise ValueError("successes must be in [0,trials] and trials positive")
    if confidence != 0.95:
        raise ValueError("the dependency-free implementation currently supports confidence=0.95 only")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == trials else min(1.0, center + radius)
    return lower, upper
