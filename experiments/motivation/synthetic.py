from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass(frozen=True)
class PathBatch:
    states: Tensor
    actions: Tensor
    valid: Tensor
    task_ids: Tensor
    episode_ids: Tensor
    success: Tensor
    failure_moments: Tensor

    def __len__(self) -> int:
        return self.states.shape[0]

    def select(self, indices: Tensor) -> PathBatch:
        return PathBatch(
            states=self.states[indices],
            actions=self.actions[indices],
            valid=self.valid[indices],
            task_ids=self.task_ids[indices],
            episode_ids=self.episode_ids[indices],
            success=self.success[indices],
            failure_moments=self.failure_moments[indices],
        )

    def model_dict(self) -> dict[str, Tensor]:
        return {
            "states": self.states,
            "actions": self.actions,
            "valid": self.valid,
            "task_ids": self.task_ids,
        }


def generate_paths(
    count: int,
    *,
    seed: int,
    domain: str,
    episode_offset: int = 0,
    num_tasks: int = 4,
) -> PathBatch:
    """Generate a source-artifact-controlled low-dimensional path diagnostic."""

    if domain not in {"expert", "current", "coarse", "perturbed"}:
        raise ValueError(f"unknown synthetic domain: {domain}")
    generator = np.random.default_rng(seed)
    horizon, state_dim, action_dim = 10, 8, 7
    task_ids = np.arange(count, dtype=np.int64) % num_tasks
    generator.shuffle(task_ids)
    states = np.zeros((count, horizon + 1, state_dim), dtype=np.float32)
    actions = np.zeros((count, horizon, action_dim), dtype=np.float32)
    states[:, 0] = generator.normal(0, 0.25, size=(count, state_dim))
    failure_onset = generator.integers(3, 8, size=count)
    if domain == "expert":
        severity = np.zeros(count, dtype=np.float32)
    elif domain == "current":
        severity = generator.choice([0.04, 0.34], size=count, p=[0.45, 0.55]).astype(np.float32)
    elif domain == "coarse":
        severity = generator.choice([0.18, 0.48], size=count, p=[0.3, 0.7]).astype(np.float32)
    else:
        severity = generator.choice([0.28, 0.55], size=count, p=[0.2, 0.8]).astype(np.float32)
    success = severity < 0.1
    failure_moments = np.zeros((count, horizon), dtype=bool)
    task_direction = np.stack(
        [np.sin(np.arange(action_dim) + task) for task in range(num_tasks)]
    ).astype(np.float32)
    task_direction /= np.linalg.norm(task_direction, axis=1, keepdims=True)
    for position in range(horizon):
        base = -0.18 * states[:, position, :action_dim]
        action_noise = generator.normal(0, 0.045, size=(count, action_dim))
        after_onset = position >= failure_onset
        drift = severity[:, None] * after_onset[:, None] * task_direction[task_ids]
        actions[:, position] = np.clip(base + action_noise + drift, -1, 1)
        transition_noise = generator.normal(0, 0.018, size=(count, state_dim))
        states[:, position + 1] = states[:, position] + transition_noise
        states[:, position + 1, :action_dim] += 0.22 * actions[:, position]
        states[:, position + 1, -1] += 0.03 * task_ids
        failure_moments[:, position] = after_onset & (severity >= 0.1)
    valid = np.ones((count, horizon), dtype=bool)
    return PathBatch(
        states=torch.from_numpy(states),
        actions=torch.from_numpy(actions),
        valid=torch.from_numpy(valid),
        task_ids=torch.from_numpy(task_ids),
        episode_ids=torch.arange(episode_offset, episode_offset + count),
        success=torch.from_numpy(success),
        failure_moments=torch.from_numpy(failure_moments),
    )


def make_splits(counts: tuple[int, int, int], *, seed: int, domain: str) -> dict[str, PathBatch]:
    names = ("train", "validation", "test")
    splits: dict[str, PathBatch] = {}
    offset = 0
    for index, (name, count) in enumerate(zip(names, counts, strict=True)):
        splits[name] = generate_paths(
            count,
            seed=seed + index * 10_007,
            domain=domain,
            episode_offset=offset,
        )
        offset += count
    identifiers = [set(batch.episode_ids.tolist()) for batch in splits.values()]
    if any(identifiers[left] & identifiers[right] for left in range(3) for right in range(left)):
        raise AssertionError("synthetic episode split leakage")
    return splits


__all__ = ["PathBatch", "generate_paths", "make_splits"]
