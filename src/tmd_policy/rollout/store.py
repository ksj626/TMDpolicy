"""Atomic versioned LIBERO replan-record storage for occupancy estimation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

ROLLOUT_SCHEMA = "tmdpolicy.libero-replans/v2"


def _sha256(value: str, field: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{field} must be a SHA-256 hex digest")


@dataclass(frozen=True)
class ReplanRecord:
    suite: str
    suite_task_id: int
    global_task_index: int
    canonical_task_uid: str
    instruction: str
    reset_seed: int
    policy_checkpoint: str
    policy_checkpoint_sha256: str
    policy_version: str
    collection_round: int
    environment_step: int
    state: Tensor
    observations: dict[str, Tensor]
    observation_metadata: dict[str, dict[str, Any]]
    planned_actions: Tensor
    executed_prefix_length: int
    executed_actions: Tensor
    terminated: bool
    truncated: bool
    success: bool
    model_revision: str
    processor_revision: str
    dataset_revision: str
    init_state_index: int | None = None

    def __post_init__(self) -> None:
        if self.suite not in {"libero_spatial", "libero_object", "libero_goal", "libero_10"}:
            raise ValueError(f"unknown LIBERO suite: {self.suite}")
        if not 0 <= self.suite_task_id < 10 or not 0 <= self.global_task_index < 40:
            raise ValueError("LIBERO task identities are outside the all-40-task contract")
        # lerobot/libero's immutable dataset task ordering is not grouped in
        # LIBERO suite order. `global_task_index` is therefore the manifest
        # identity resolved from the instruction, not suite_offset+local_id.
        expected_uid_prefix = f"libero:{self.global_task_index:02d}:"
        if not self.canonical_task_uid.startswith(expected_uid_prefix):
            raise ValueError(
                "canonical LIBERO task UID disagrees with the dataset-global task identity"
            )
        if not self.canonical_task_uid or not self.instruction:
            raise ValueError("canonical task UID and instruction must be nonempty")
        if self.init_state_index is not None and self.init_state_index < 0:
            raise ValueError("LIBERO init-state index must be nonnegative")
        if self.state.shape != (8,) or not torch.isfinite(self.state).all():
            raise ValueError("replan-start state must be finite [8]")
        if self.planned_actions.shape != (50, 7) or not torch.isfinite(self.planned_actions).all():
            raise ValueError("full canonical planned action chunk must be finite [50,7]")
        if not 0 <= self.executed_prefix_length <= 50:
            raise ValueError("executed prefix length must be in [0,50]")
        if self.executed_actions.shape != (self.executed_prefix_length, 7):
            raise ValueError("executed actions must exactly match the executed prefix length")
        if not torch.equal(
            self.executed_actions.float(), self.planned_actions[: self.executed_prefix_length].float()
        ):
            raise ValueError("executed actions are not the recorded plan prefix")
        if not self.observations:
            raise ValueError("a replan record must retain canonical camera observations")
        if set(self.observations) != set(self.observation_metadata):
            raise ValueError("observation metadata must cover every stored camera")
        for key, image in self.observations.items():
            metadata = self.observation_metadata[key]
            if list(image.shape) != list(metadata.get("shape", [])):
                raise ValueError(f"observation shape metadata mismatch for {key}")
            if str(image.dtype) != metadata.get("dtype"):
                raise ValueError(f"observation dtype metadata mismatch for {key}")
            if image.dtype.is_floating_point and not torch.isfinite(image).all():
                raise ValueError(f"observation contains non-finite pixels: {key}")
        _sha256(self.policy_checkpoint_sha256, "policy_checkpoint_sha256")
        for revision in (self.model_revision, self.processor_revision, self.dataset_revision):
            if len(revision) != 40:
                raise ValueError("model/processor/dataset revisions must be immutable Hub commits")


@dataclass(frozen=True)
class RolloutEpisode:
    replans: tuple[ReplanRecord, ...]
    split: Literal["train", "validation", "test"]

    def __post_init__(self) -> None:
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown rollout split: {self.split}")
        if not self.replans:
            raise ValueError("rollout episode must contain at least one real replan record")
        first = self.replans[0]
        identity = (
            first.suite,
            first.suite_task_id,
            first.global_task_index,
            first.canonical_task_uid,
            first.instruction,
            first.reset_seed,
            first.init_state_index,
            first.policy_checkpoint,
            first.policy_checkpoint_sha256,
            first.policy_version,
            first.collection_round,
            first.model_revision,
            first.processor_revision,
            first.dataset_revision,
        )
        if any(
            (
                value.suite,
                value.suite_task_id,
                value.global_task_index,
                value.canonical_task_uid,
                value.instruction,
                value.reset_seed,
                value.init_state_index,
                value.policy_checkpoint,
                value.policy_checkpoint_sha256,
                value.policy_version,
                value.collection_round,
                value.model_revision,
                value.processor_revision,
                value.dataset_revision,
            )
            != identity
            for value in self.replans
        ):
            raise ValueError("all replans in an episode must share suite/task/reset identity")
        actual_steps = [value.environment_step for value in self.replans]
        for left, right in zip(self.replans, self.replans[1:], strict=False):
            if right.environment_step != left.environment_step + left.executed_prefix_length:
                raise ValueError("replan steps must advance by the actual executed prefix length")
        if any(value.terminated or value.truncated for value in self.replans[:-1]):
            raise ValueError("only the final replan may terminate or truncate an episode")


class RolloutStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.episodes = self.root / "episodes"
        self.index_path = self.root / "episodes.json"

    def initialize(self, metadata: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        self.episodes.mkdir()
        manifest = {"schema": ROLLOUT_SCHEMA, **metadata}
        target = self.root / "manifest.json"
        temporary = target.with_suffix(".partial")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, target)
        self._write_index([])

    def _manifest(self) -> dict[str, Any]:
        path = self.root / "manifest.json"
        if not path.exists():
            raise RuntimeError("rollout store must be initialized before use")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema") != ROLLOUT_SCHEMA:
            raise ValueError(
                f"unsupported rollout schema {manifest.get('schema')!r}; v1 episode windows cannot "
                "be reinterpreted as v2 replan records and must be recollected"
            )
        return manifest

    def _write_index(self, rows: list[dict[str, Any]]) -> None:
        temporary = self.index_path.with_suffix(".partial")
        temporary.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.index_path)

    @staticmethod
    def _record_payload(record: ReplanRecord) -> dict[str, Any]:
        return {
            "suite": record.suite,
            "suite_task_id": record.suite_task_id,
            "global_task_index": record.global_task_index,
            "canonical_task_uid": record.canonical_task_uid,
            "instruction": record.instruction,
            "reset_seed": record.reset_seed,
            "init_state_index": record.init_state_index,
            "policy_checkpoint": record.policy_checkpoint,
            "policy_checkpoint_sha256": record.policy_checkpoint_sha256,
            "policy_version": record.policy_version,
            "collection_round": record.collection_round,
            "environment_step": record.environment_step,
            "state": record.state.cpu(),
            "observations": {key: value.cpu() for key, value in record.observations.items()},
            "observation_metadata": record.observation_metadata,
            "planned_actions": record.planned_actions.cpu(),
            "executed_prefix_length": record.executed_prefix_length,
            "executed_actions": record.executed_actions.cpu(),
            "terminated": record.terminated,
            "truncated": record.truncated,
            "success": record.success,
            "model_revision": record.model_revision,
            "processor_revision": record.processor_revision,
            "dataset_revision": record.dataset_revision,
        }

    def append(self, episode: RolloutEpisode) -> dict[str, Any]:
        self._manifest()
        rows = self.records()
        episode_id = len(rows)
        filename = f"episode-{episode_id:06d}.pt"
        target = self.episodes / filename
        temporary = target.with_suffix(".partial")
        torch.save({"replans": [self._record_payload(value) for value in episode.replans]}, temporary)
        os.replace(temporary, target)
        first, last = episode.replans[0], episode.replans[-1]
        row = {
            "episode_id": episode_id,
            "payload": f"episodes/{filename}",
            "replans": len(episode.replans),
            "executed_steps": sum(value.executed_prefix_length for value in episode.replans),
            "suite": first.suite,
            "suite_task_id": first.suite_task_id,
            "global_task_index": first.global_task_index,
            "canonical_task_uid": first.canonical_task_uid,
            "instruction": first.instruction,
            "reset_seed": first.reset_seed,
            "init_state_index": first.init_state_index,
            "success": last.success,
            "terminated": last.terminated,
            "truncated": last.truncated,
            "policy_checkpoint": first.policy_checkpoint,
            "policy_checkpoint_sha256": first.policy_checkpoint_sha256,
            "policy_version": first.policy_version,
            "collection_round": first.collection_round,
            "split": episode.split,
        }
        rows.append(row)
        self._write_index(rows)
        return row

    def records(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        value = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("rollout index must be a JSON list")
        return value

    def load_replans(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        self._manifest()
        payload = torch.load(self.root / row["payload"], map_location="cpu", weights_only=True)
        replans = payload.get("replans")
        if not isinstance(replans, list) or len(replans) != row["replans"]:
            raise ValueError(f"corrupt rollout payload: {row['payload']}")
        return replans

    def validate(self) -> dict[str, Any]:
        manifest = self._manifest()
        rows = self.records()
        support: set[int] = set()
        replans = steps = 0
        for row in rows:
            values = self.load_replans(row)
            episode = RolloutEpisode(
                replans=tuple(ReplanRecord(**value) for value in values),
                split=row["split"],
            )
            replans += len(values)
            steps += sum(int(value["executed_prefix_length"]) for value in values)
            support.add(int(row["global_task_index"]))
            for value in values:
                if value["planned_actions"].shape != (50, 7):
                    raise ValueError(f"corrupt full plan in {row['payload']}")
                if value["executed_actions"].shape != (value["executed_prefix_length"], 7):
                    raise ValueError(f"corrupt executed prefix in {row['payload']}")
            first = episode.replans[0]
            if (row["suite"], row["suite_task_id"], row["global_task_index"]) != (
                first.suite,
                first.suite_task_id,
                first.global_task_index,
            ):
                raise ValueError(f"rollout index identity differs from {row['payload']}")
        return {
            "episodes": len(rows),
            "replans": replans,
            "steps": steps,
            "task_support": sorted(support),
            "manifest": manifest,
        }


__all__ = ["ROLLOUT_SCHEMA", "ReplanRecord", "RolloutEpisode", "RolloutStore"]
