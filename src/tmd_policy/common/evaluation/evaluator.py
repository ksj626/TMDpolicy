from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import wilson_interval
from .outcomes import EpisodeOutcome


@dataclass(frozen=True)
class EpisodeEvaluation:
    canonical_task_uid: str
    reset_seed: int
    outcome: EpisodeOutcome
    executed_actions: np.ndarray
    preprocessing_latency_s: tuple[float, ...]
    model_latency_s: tuple[float, ...]
    postprocessing_latency_s: tuple[float, ...]
    environment_latency_s: tuple[float, ...]
    end_to_end_episode_latency_s: float
    peak_allocated_memory_bytes: int | None

    def __post_init__(self) -> None:
        if self.executed_actions.ndim != 2 or not np.isfinite(self.executed_actions).all():
            raise ValueError("executed actions must be finite [steps,action_dim]")
        if self.reset_seed < 0 or self.end_to_end_episode_latency_s < 0:
            raise ValueError("evaluation seed/latency must be nonnegative")
        lengths = {
            len(self.preprocessing_latency_s), len(self.model_latency_s),
            len(self.postprocessing_latency_s), len(self.environment_latency_s),
        }
        if len(lengths) != 1:
            raise ValueError("per-replan latency components must align")


def _latency(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean_s": float("nan"), "p50_s": float("nan"), "p95_s": float("nan")}
    return {
        "mean_s": float(array.mean()),
        "p50_s": float(np.quantile(array, 0.5)),
        "p95_s": float(np.quantile(array, 0.95)),
    }


def summarize_policy_evaluation(
    episodes: list[EpisodeEvaluation], *, held_out_diagnostics: dict[str, Any] | None = None
) -> dict[str, Any]:
    if not episodes:
        raise ValueError("policy evaluation needs complete episodes")
    by_task: dict[str, list[EpisodeEvaluation]] = defaultdict(list)
    for episode in episodes:
        by_task[episode.canonical_task_uid].append(episode)
    task_reports = {}
    for task, rows in sorted(by_task.items()):
        successes = sum(row.outcome.task_success for row in rows)
        task_reports[task] = {
            "successes": successes, "episodes": len(rows),
            "success_rate": successes / len(rows), "wilson_95": wilson_interval(successes, len(rows)),
        }
    actions = np.concatenate([episode.executed_actions for episode in episodes], axis=0)
    per_episode_differences = [
        np.diff(episode.executed_actions, axis=0)
        for episode in episodes
        if len(episode.executed_actions) > 1
    ]
    differences = (
        np.concatenate(per_episode_differences, axis=0)
        if per_episode_differences
        else np.zeros((1, actions.shape[1]), dtype=actions.dtype)
    )
    model_latencies = [value for episode in episodes for value in episode.model_latency_s]
    successes = sum(row["successes"] for row in task_reports.values())
    return {
        "per_task": task_reports,
        "macro_success": float(np.mean([row["success_rate"] for row in task_reports.values()])),
        "micro_success": successes / len(episodes),
        "micro_wilson_95": wilson_interval(successes, len(episodes)),
        "cold_model_latency_s": model_latencies[0] if model_latencies else float("nan"),
        "warm_model_latency": _latency(model_latencies[1:]),
        "preprocessing_latency": _latency([v for row in episodes for v in row.preprocessing_latency_s]),
        "postprocessing_latency": _latency([v for row in episodes for v in row.postprocessing_latency_s]),
        "environment_latency": _latency([v for row in episodes for v in row.environment_latency_s]),
        "episode_latency": _latency([row.end_to_end_episode_latency_s for row in episodes]),
        "action_diversity_mean_std": float(actions.std(axis=0).mean()),
        "action_smoothness_mean_l2_delta": float(np.linalg.norm(differences, axis=1).mean()),
        "peak_allocated_memory_bytes": max((row.peak_allocated_memory_bytes or 0 for row in episodes), default=0),
        "memory_protocol": (
            "reset torch CUDA peak memory immediately before synchronized evaluation; "
            "read max_memory_allocated after the final synchronized episode"
        ),
        "held_out_diagnostics": held_out_diagnostics or {},
    }
