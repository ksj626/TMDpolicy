from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = 2
PLAN_HORIZON = 50
EXECUTION_HORIZON = 10
CANONICAL_ACTION_DIM = 7
CANONICAL_STATE_DIM = 8


def _array(value: Any, dtype: np.dtype | type) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _boolean_mask(name: str, value: Any, length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.bool_:
        raise TypeError(f"{name} must have boolean dtype, got {raw.dtype}")
    mask = raw.astype(bool, copy=False)
    if mask.shape != (length,):
        raise ValueError(f"{name} expected shape {(length,)}, got {mask.shape}")
    if np.any(mask[1:] & ~mask[:-1]):
        raise ValueError(f"{name} must be prefix-contiguous (true values before false values)")
    return mask


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} expected shape {shape}, got {value.shape}")


def _finite(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")


def _canonical_actions(name: str, value: Any, shape: tuple[int, int]) -> np.ndarray:
    actions = _array(value, np.float32)
    _require_shape(name, actions, shape)
    _finite(name, actions)
    if actions.size and (float(actions.min()) < -1.000001 or float(actions.max()) > 1.000001):
        raise ValueError(f"{name} must be in canonical LIBERO bounds [-1, 1]")
    return actions


def _states(name: str, value: Any, shape: tuple[int, int]) -> np.ndarray:
    states = _array(value, np.float32)
    _require_shape(name, states, shape)
    _finite(name, states)
    return states


def _identity(sample_id: str, observation_id: str, task_index: int, instruction: str) -> None:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", sample_id) is None:
        raise ValueError("sample_id must be immutable/filesystem-safe")
    if not observation_id.strip() or task_index < 0 or not instruction.strip():
        raise ValueError("observation_id, nonnegative task_index, and instruction are required")


@dataclass
class ExpertChunk:
    """Versioned expert record with a padded complete 10-transition path."""

    sample_id: str
    observation_id: str
    dataset_id: str
    dataset_revision: str
    episode_index: int
    task_index: int
    instruction: str
    start_frame: int
    plan_actions: np.ndarray
    plan_valid: np.ndarray
    path_states: np.ndarray
    path_actions: np.ndarray
    path_valid: np.ndarray
    split: str = "train"
    images: dict[str, np.ndarray] = field(default_factory=dict)
    reset_seed: int | None = None
    is_episode_start: bool = False
    reaches_episode_end: bool = False

    def __post_init__(self) -> None:
        _identity(self.sample_id, self.observation_id, self.task_index, self.instruction)
        if not self.dataset_id.strip() or not self.dataset_revision.strip():
            raise ValueError("dataset ID and immutable revision are required")
        if self.episode_index < 0 or self.start_frame < 0:
            raise ValueError("episode_index and start_frame must be nonnegative")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("expert split must be train, validation, or test")
        self.plan_actions = _canonical_actions(
            "plan_actions", self.plan_actions, (PLAN_HORIZON, CANONICAL_ACTION_DIM)
        )
        self.plan_valid = _boolean_mask("plan_valid", self.plan_valid, PLAN_HORIZON)
        self.path_actions = _canonical_actions(
            "path_actions", self.path_actions, (EXECUTION_HORIZON, CANONICAL_ACTION_DIM)
        )
        self.path_valid = _boolean_mask("path_valid", self.path_valid, EXECUTION_HORIZON)
        self.path_states = _states(
            "path_states", self.path_states, (EXECUTION_HORIZON + 1, CANONICAL_STATE_DIM)
        )
        if np.any(self.path_valid & ~self.plan_valid[:EXECUTION_HORIZON]):
            raise ValueError("path_valid cannot mark a transition invalid in plan_valid as real")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "expert",
            "sample_id": self.sample_id,
            "observation_id": self.observation_id,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "episode_index": self.episode_index,
            "task_index": self.task_index,
            "instruction": self.instruction,
            "start_frame": self.start_frame,
            "reset_seed": self.reset_seed,
            "is_episode_start": self.is_episode_start,
            "reaches_episode_end": self.reaches_episode_end,
            "split": self.split,
        }

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        result = {
            "plan_actions": self.plan_actions,
            "plan_valid": self.plan_valid,
            "path_states": self.path_states,
            "path_actions": self.path_actions,
            "path_valid": self.path_valid,
        }
        result.update({f"image::{key}": np.asarray(value) for key, value in self.images.items()})
        return result


@dataclass
class RolloutChunk:
    """A real executed prefix of a 50-action student plan."""

    sample_id: str
    observation_id: str
    policy_checkpoint: str
    policy_version: str
    collection_round: int
    task_index: int
    instruction: str
    chunk_index: int
    plan_actions: np.ndarray
    executed_actions: np.ndarray
    path_states: np.ndarray
    path_valid: np.ndarray
    success: bool
    terminated: bool
    truncated: bool
    reset_seed: int
    outer_noise_seed: int
    inner_noise_seeds: tuple[int, ...]
    preprocessing_latency_s: float = 0.0
    model_latency_s: float = 0.0
    postprocessing_latency_s: float = 0.0
    environment_latency_s: float = 0.0
    chunk_start_images: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _identity(self.sample_id, self.observation_id, self.task_index, self.instruction)
        if (
            not self.policy_checkpoint.strip()
            or not self.policy_version.strip()
            or self.collection_round < 0
            or self.chunk_index < 0
            or self.reset_seed < 0
            or self.outer_noise_seed < 0
        ):
            raise ValueError("complete nonnegative rollout and policy provenance is required")
        self.plan_actions = _canonical_actions(
            "plan_actions", self.plan_actions, (PLAN_HORIZON, CANONICAL_ACTION_DIM)
        )
        executed = np.asarray(self.executed_actions)
        if executed.ndim != 2 or executed.shape[1:] != (CANONICAL_ACTION_DIM,):
            raise ValueError(f"executed_actions expected [0..10,7], got {executed.shape}")
        if executed.shape[0] > EXECUTION_HORIZON:
            raise ValueError("a rollout chunk cannot execute more than 10 actions")
        self.executed_actions = _canonical_actions(
            "executed_actions", executed, (executed.shape[0], CANONICAL_ACTION_DIM)
        )
        self.path_states = _states(
            "path_states",
            self.path_states,
            (self.executed_actions.shape[0] + 1, CANONICAL_STATE_DIM),
        )
        self.path_valid = _boolean_mask(
            "path_valid", self.path_valid, self.executed_actions.shape[0]
        )
        if not self.path_valid.all():
            raise ValueError("rollout storage contains only real transitions, so path_valid must be all true")
        seeds = tuple(int(seed) for seed in self.inner_noise_seeds)
        if seeds and min(seeds) < 0:
            raise ValueError("inner-noise seeds must be nonnegative")
        self.inner_noise_seeds = seeds
        for name in (
            "preprocessing_latency_s",
            "model_latency_s",
            "postprocessing_latency_s",
            "environment_latency_s",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative duration")

    @property
    def policy_noise_seed(self) -> int:
        """Compatibility spelling for pre-v2 callers; new records store outer_noise_seed."""

        return self.outer_noise_seed

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "rollout",
            "sample_id": self.sample_id,
            "observation_id": self.observation_id,
            "policy_checkpoint": self.policy_checkpoint,
            "policy_version": self.policy_version,
            "collection_round": self.collection_round,
            "task_index": self.task_index,
            "instruction": self.instruction,
            "chunk_index": self.chunk_index,
            "success": self.success,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "reset_seed": self.reset_seed,
            "outer_noise_seed": self.outer_noise_seed,
            "inner_noise_seeds": list(self.inner_noise_seeds),
            "preprocessing_latency_s": self.preprocessing_latency_s,
            "model_latency_s": self.model_latency_s,
            "postprocessing_latency_s": self.postprocessing_latency_s,
            "environment_latency_s": self.environment_latency_s,
        }

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        result = {
            "plan_actions": self.plan_actions,
            "executed_actions": self.executed_actions,
            "path_states": self.path_states,
            "path_valid": self.path_valid,
        }
        result.update(
            {f"image::{key}": np.asarray(value) for key, value in self.chunk_start_images.items()}
        )
        return result


@dataclass
class TeacherQuery:
    """Immutable, postprocessed teacher action chunk at a stored observation."""

    sample_id: str
    observation_id: str
    task_index: int
    instruction: str
    teacher_checkpoint: str
    teacher_revision: str
    processor_revision: str
    sampling_seed: int
    inference_steps: int
    sample_index: int
    action_chunk: np.ndarray
    action_valid: np.ndarray

    def __post_init__(self) -> None:
        _identity(self.sample_id, self.observation_id, self.task_index, self.instruction)
        if not self.teacher_checkpoint.strip() or not self.teacher_revision.strip():
            raise ValueError("teacher checkpoint and revision are required")
        if not self.processor_revision.strip():
            raise ValueError("teacher processor revision is required")
        if self.sampling_seed < 0 or self.inference_steps < 1 or self.sample_index < 0:
            raise ValueError("teacher sampling provenance must be nonnegative and steps positive")
        self.action_chunk = _canonical_actions(
            "action_chunk", self.action_chunk, (PLAN_HORIZON, CANONICAL_ACTION_DIM)
        )
        self.action_valid = _boolean_mask("action_valid", self.action_valid, PLAN_HORIZON)

    @staticmethod
    def make_cache_key(
        observation_id: str,
        teacher_checkpoint: str,
        teacher_revision: str,
        processor_revision: str,
        inference_steps: int,
        sampling_seed: int,
        sample_index: int,
    ) -> str:
        identity = {
            "inference_steps": inference_steps,
            "observation_id": observation_id,
            "processor_revision": processor_revision,
            "sample_index": sample_index,
            "sampling_seed": sampling_seed,
            "teacher_checkpoint": teacher_checkpoint,
            "teacher_revision": teacher_revision,
        }
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @property
    def cache_key(self) -> str:
        return self.make_cache_key(
            self.observation_id,
            self.teacher_checkpoint,
            self.teacher_revision,
            self.processor_revision,
            self.inference_steps,
            self.sampling_seed,
            self.sample_index,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "teacher_query",
            "sample_id": self.sample_id,
            "observation_id": self.observation_id,
            "task_index": self.task_index,
            "instruction": self.instruction,
            "teacher_checkpoint": self.teacher_checkpoint,
            "teacher_revision": self.teacher_revision,
            "processor_revision": self.processor_revision,
            "sampling_seed": self.sampling_seed,
            "inference_steps": self.inference_steps,
            "sample_index": self.sample_index,
            "cache_key": self.cache_key,
        }

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {"action_chunk": self.action_chunk, "action_valid": self.action_valid}


__all__ = [
    "CANONICAL_ACTION_DIM",
    "CANONICAL_STATE_DIM",
    "EXECUTION_HORIZON",
    "PLAN_HORIZON",
    "SCHEMA_VERSION",
    "ExpertChunk",
    "RolloutChunk",
    "TeacherQuery",
]
