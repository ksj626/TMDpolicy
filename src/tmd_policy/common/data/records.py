from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tmd_policy.common.tasks import TaskIdentity


def validate_image(name: str, value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim not in {3, 4}:
        raise ValueError(f"{name} must be CHW/HWC or a batch thereof, got {image.shape}")
    channel_axes = [index for index, size in enumerate(image.shape[-3:]) if size in {1, 3, 4}]
    if not channel_axes:
        raise ValueError(f"{name} has no identifiable 1/3/4-channel image axis: {image.shape}")
    if image.dtype == np.uint8:
        return image
    if image.dtype not in {np.dtype(np.float16), np.dtype(np.float32), np.dtype(np.float64)}:
        raise TypeError(f"{name} must be uint8 or floating point, got {image.dtype}")
    if not np.isfinite(image).all():
        raise ValueError(f"{name} contains NaN or infinity")
    minimum, maximum = float(image.min(initial=0)), float(image.max(initial=0))
    if minimum < 0 or maximum > 1:
        raise ValueError(f"{name} floating images must be in [0,1], got [{minimum},{maximum}]")
    return image


def _array_digest(hasher: Any, key: str, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    hasher.update(key.encode("utf-8"))
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(json.dumps(contiguous.shape).encode("ascii"))
    hasher.update(contiguous.tobytes())


@dataclass
class ResearchRecord:
    kind: str
    sample_id: str
    task: TaskIdentity
    episode_index: int
    frame_index: int
    split: str
    arrays: dict[str, np.ndarray]
    metadata_extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in {"expert", "student_rollout", "teacher_label", "dmd2_generated"}:
            raise ValueError(f"unsupported record kind: {self.kind}")
        if not self.sample_id or min(self.episode_index, self.frame_index) < 0:
            raise ValueError("record identity and nonnegative episode/frame are required")
        if self.split not in {"train", "validation", "test", "online"}:
            raise ValueError(f"invalid split: {self.split}")
        for name, value in tuple(self.arrays.items()):
            array = np.asarray(value)
            if array.dtype == object:
                raise TypeError(f"object arrays are forbidden: {name}")
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                raise ValueError(f"{name} contains NaN or infinity")
            if name.startswith("image::"):
                array = validate_image(name, array)
            self.arrays[name] = array

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "kind": self.kind,
            "sample_id": self.sample_id,
            "canonical_task_uid": self.task.canonical_task_uid,
            "task_identity": self.task.to_dict(),
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "split": self.split,
            **self.metadata_extra,
            "content_hash": self.content_hash,
        }

    @property
    def content_hash(self) -> str:
        hasher = hashlib.sha256()
        stable = {
            "schema_version": 3,
            "kind": self.kind,
            "sample_id": self.sample_id,
            "task_identity": self.task.to_dict(),
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "split": self.split,
            "metadata_extra": self.metadata_extra,
        }
        hasher.update(json.dumps(stable, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        for key in sorted(self.arrays):
            _array_digest(hasher, key, self.arrays[key])
        return hasher.hexdigest()


def _require_keys(label: str, values: dict[str, Any], required: set[str]) -> None:
    missing = sorted(required - set(values))
    if missing:
        raise ValueError(f"{label} record is missing required fields: {missing}")


@dataclass
class ExpertResearchRecord(ResearchRecord):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "expert":
            raise ValueError("ExpertResearchRecord kind must be expert")
        _require_keys(
            "expert arrays",
            self.arrays,
            {
                "action_plan_original", "action_plan_canonical", "action_valid",
                "state_sequence", "executed_path_actions", "path_valid",
            },
        )
        _require_keys(
            "expert metadata", self.metadata_extra,
            {"observation_id", "processor_revision", "normalizer_revision"},
        )
        if self.arrays["action_valid"].dtype != np.bool_ or not self.arrays["action_valid"].any():
            raise ValueError("expert action_valid must be boolean and nonempty")
        if self.arrays["path_valid"].dtype != np.bool_ or not self.arrays["path_valid"].any():
            raise ValueError("expert path_valid must be boolean and nonempty")


@dataclass
class StudentRolloutRecord(ResearchRecord):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "student_rollout" or self.split != "online":
            raise ValueError("student rollouts must have kind=student_rollout and split=online")
        _require_keys(
            "student rollout arrays", self.arrays,
            {"pre_action_state", "sampled_action_chunk", "executed_actions", "resulting_states", "path_valid"},
        )
        _require_keys(
            "student rollout metadata", self.metadata_extra,
            {
                "observation_id", "policy_checkpoint", "policy_version", "collection_round",
                "reset_seed", "environment_seed", "outer_noise_seed", "inner_noise_seeds",
                "success", "terminated", "environment_truncated", "local_time_limit",
                "preprocessing_latency_s", "model_latency_s", "postprocessing_latency_s",
                "environment_latency_s",
            },
        )
        actions = self.arrays["executed_actions"]
        states = self.arrays["resulting_states"]
        mask = self.arrays["path_valid"]
        if states.shape[0] != actions.shape[0] + 1 or mask.shape != (actions.shape[0],):
            raise ValueError("student rollout must align L actions, L+1 states, and L validity flags")
        if mask.dtype != np.bool_ or not mask.all():
            raise ValueError("stored rollout prefixes contain only real boolean-valid transitions")


@dataclass
class TeacherLabelRecord(ResearchRecord):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "teacher_label":
            raise ValueError("TeacherLabelRecord kind must be teacher_label")
        _require_keys("teacher arrays", self.arrays, {"teacher_samples", "evaluated_student_action"})
        _require_keys(
            "teacher metadata", self.metadata_extra,
            {
                "observation_id", "teacher_model_revision", "teacher_processor_revision",
                "teacher_inference_schedule", "query_seed", "sample_index", "cache_key",
            },
        )


@dataclass
class DMD2GeneratedRecord(ResearchRecord):
    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "dmd2_generated":
            raise ValueError("DMD2GeneratedRecord kind must be dmd2_generated")
        _require_keys("DMD2 arrays", self.arrays, {"generated_action_chunk", "generation_noise"})
        _require_keys(
            "DMD2 metadata", self.metadata_extra,
            {
                "observation_id", "student_policy_version", "generation_schedule",
                "fake_score_version", "teacher_query_identity", "gan_real_identity",
            },
        )
