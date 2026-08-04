from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np

from tmd_policy.common.evaluation import average_precision, precision_recall_auc


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return math.nan
    ranks = _average_ranks(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(labels: Any, logits: Any, bins: int = 10) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if labels.shape != logits.shape or not len(labels):
        raise ValueError("nonempty labels and logits must have identical shapes")
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    predictions = probabilities >= 0.5
    bce = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in pairwise(edges):
        mask = (probabilities >= low) & (probabilities < high if high < 1 else probabilities <= high)
        if mask.any():
            ece += float(mask.mean()) * abs(float(probabilities[mask].mean() - labels[mask].mean()))
    return {
        "bce": float(bce.mean()),
        "accuracy": float((predictions == labels).mean()),
        "roc_auc": _roc_auc(labels, logits),
        "average_precision": average_precision(labels, logits),
        "pr_auc": precision_recall_auc(labels, logits),
        "brier": float(np.square(probabilities - labels).mean()),
        "ece": ece,
        "saturation_fraction": float(((probabilities < 0.01) | (probabilities > 0.99)).mean()),
        "expert_logit_mean": float(logits[labels == 1].mean()) if (labels == 1).any() else math.nan,
        "student_logit_mean": float(logits[labels == 0].mean()) if (labels == 0).any() else math.nan,
    }


def discriminator_report(expert_logits: Any, student_logits: Any) -> dict[str, float]:
    expert = np.asarray(expert_logits).reshape(-1)
    student = np.asarray(student_logits).reshape(-1)
    labels = np.concatenate((np.ones_like(expert, dtype=int), np.zeros_like(student, dtype=int)))
    return binary_metrics(labels, np.concatenate((expert, student)))


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    finite = np.isfinite(left) & np.isfinite(right)
    left, right = left[finite], right[finite]
    if len(left) < 2 or left.std() == 0 or right.std() == 0:
        return math.nan
    return float(np.corrcoef(left, right)[0, 1])


def prefix_discriminator_report(
    expert_logits: Any,
    student_logits: Any,
    expert_valid: Any,
    student_valid: Any,
    *,
    student_success: Any | None = None,
    student_failure_moments: Any | None = None,
    expert_task_ids: Any | None = None,
    student_task_ids: Any | None = None,
) -> dict[str, Any]:
    """Metrics required for held-out causal-prefix diagnostics.

    `student_failure_moments` is a boolean `(batch, horizon)` annotation. Its
    correlation uses negative logit increments, so larger values mean a stronger
    mismatch localized at that transition.
    """
    expert = np.asarray(expert_logits, dtype=np.float64)
    student = np.asarray(student_logits, dtype=np.float64)
    expert_mask = np.asarray(expert_valid, dtype=bool)
    student_mask = np.asarray(student_valid, dtype=bool)
    if expert.shape != expert_mask.shape or student.shape != student_mask.shape:
        raise ValueError("logit and validity shapes must match")
    if expert.shape[1] != student.shape[1]:
        raise ValueError("expert and student horizons must match")
    result: dict[str, Any] = {
        "overall": discriminator_report(expert[expert_mask], student[student_mask]),
        "prefix_by_position": [],
        "expert_logit_quantiles": np.quantile(expert[expert_mask], [0.01, 0.5, 0.99]).tolist(),
        "student_logit_quantiles": np.quantile(student[student_mask], [0.01, 0.5, 0.99]).tolist(),
    }
    for position in range(expert.shape[1]):
        e_values = expert[:, position][expert_mask[:, position]]
        s_values = student[:, position][student_mask[:, position]]
        metrics = discriminator_report(e_values, s_values) if len(e_values) and len(s_values) else {}
        result["prefix_by_position"].append({"position": position, **metrics})
    student_last = student_mask.sum(axis=1).clip(min=1) - 1
    final = student[np.arange(len(student)), student_last]
    if student_success is not None:
        result["final_logit_success_correlation"] = _correlation(final, np.asarray(student_success))
    if student_failure_moments is not None:
        failures = np.asarray(student_failure_moments, dtype=bool)
        if failures.shape != student.shape:
            raise ValueError("failure moments must have the student prefix shape")
        increments = np.diff(np.concatenate((np.zeros((len(student), 1)), student), axis=1), axis=1)
        result["negative_increment_failure_correlation"] = _correlation(
            -increments[student_mask], failures[student_mask]
        )
    if expert_task_ids is not None and student_task_ids is not None:
        expert_tasks = np.asarray(expert_task_ids).reshape(-1)
        student_tasks = np.asarray(student_task_ids).reshape(-1)
        by_task: dict[str, Any] = {}
        for task in np.union1d(expert_tasks, student_tasks):
            e_rows = expert_tasks == task
            s_rows = student_tasks == task
            if e_rows.any() and s_rows.any():
                by_task[str(int(task))] = discriminator_report(
                    expert[e_rows][expert_mask[e_rows]], student[s_rows][student_mask[s_rows]]
                )
        result["by_task"] = by_task
    return result


def bootstrap_episode_statistic(
    values: Any,
    *,
    statistic: Any = np.mean,
    task_ids: Any | None = None,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Task-stratified confidence interval resampling complete episode rows."""

    episode_values = np.asarray(values)
    if episode_values.ndim == 0 or len(episode_values) < 1:
        raise ValueError("values must contain at least one complete episode")
    if not 0 < confidence < 1 or resamples < 1:
        raise ValueError("confidence must be in (0,1) and resamples positive")
    if task_ids is None:
        tasks = np.zeros(len(episode_values), dtype=np.int64)
    else:
        tasks = np.asarray(task_ids).reshape(-1)
        if tasks.shape != (len(episode_values),):
            raise ValueError("task_ids must contain one task per episode")
    generator = np.random.default_rng(seed)
    groups = [np.flatnonzero(tasks == task) for task in np.unique(tasks)]
    estimates = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        sampled = np.concatenate(
            [generator.choice(group, size=len(group), replace=True) for group in groups]
        )
        estimates[index] = float(statistic(episode_values[sampled]))
    tail = (1.0 - confidence) / 2.0
    return {
        "estimate": float(statistic(episode_values)),
        "ci_lower": float(np.quantile(estimates, tail)),
        "ci_upper": float(np.quantile(estimates, 1.0 - tail)),
        "confidence": confidence,
        "episodes": len(episode_values),
        "resamples": resamples,
        "seed": seed,
    }
