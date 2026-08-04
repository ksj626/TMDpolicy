"""Versioned complete-episode student rollout storage."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

ROLLOUT_SCHEMA = "tmdpolicy.libero-rollout/v1"


@dataclass(frozen=True)
class RolloutEpisode:
    states: Tensor
    actions: Tensor
    visual: Tensor
    task_index: int
    canonical_task_uid: str
    instruction: str
    reset_seed: int
    success: bool
    terminated: bool
    truncated: bool
    policy_checkpoint: str
    policy_checkpoint_sha256: str
    collection_round: int
    split: Literal["train", "validation", "test"]

    def __post_init__(self) -> None:
        length = self.actions.shape[0]
        if self.states.shape != (length + 1, 8):
            raise ValueError("rollout states must be [T+1,8]")
        if self.actions.shape[1:] != (7,) or self.visual.shape != (length, 6):
            raise ValueError("rollout actions/visual must be [T,7]/[T,6]")
        for value in (self.states, self.actions, self.visual):
            if not torch.isfinite(value).all():
                raise ValueError("rollout tensors must be finite")
        if length < 1 or self.reset_seed < 0 or self.collection_round < 0:
            raise ValueError("rollout length/seed/round must be valid")
        if len(self.policy_checkpoint_sha256) != 64:
            raise ValueError("rollout must record a SHA-256 producing checkpoint identity")


class RolloutStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.episodes = self.root / "episodes"
        self.index_path = self.root / "episodes.jsonl"

    def initialize(self, metadata: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=False)
        self.episodes.mkdir()
        manifest = {"schema": ROLLOUT_SCHEMA, **metadata}
        (self.root / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def append(self, episode: RolloutEpisode) -> dict[str, Any]:
        if not (self.root / "manifest.json").exists():
            raise RuntimeError("rollout store must be initialized before append")
        episode_id = sum(1 for _ in self.records())
        filename = f"episode-{episode_id:06d}.pt"
        target = self.episodes / filename
        temporary = target.with_suffix(".partial")
        torch.save(
            {"states": episode.states.cpu(), "actions": episode.actions.cpu(), "visual": episode.visual.cpu()},
            temporary,
        )
        os.replace(temporary, target)
        row = {
            "episode_id": episode_id,
            "payload": f"episodes/{filename}",
            "length": int(episode.actions.shape[0]),
            "task_index": episode.task_index,
            "canonical_task_uid": episode.canonical_task_uid,
            "instruction": episode.instruction,
            "reset_seed": episode.reset_seed,
            "success": episode.success,
            "terminated": episode.terminated,
            "truncated": episode.truncated,
            "policy_checkpoint": episode.policy_checkpoint,
            "policy_checkpoint_sha256": episode.policy_checkpoint_sha256,
            "collection_round": episode.collection_round,
            "split": episode.split,
        }
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        return row

    def records(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        return [json.loads(line) for line in self.index_path.read_text(encoding="utf-8").splitlines() if line]

    def validate(self) -> dict[str, Any]:
        manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != ROLLOUT_SCHEMA:
            raise ValueError("unsupported rollout store schema")
        rows = self.records()
        for row in rows:
            payload = torch.load(self.root / row["payload"], map_location="cpu", weights_only=True)
            if payload["actions"].shape != (row["length"], 7):
                raise ValueError(f"corrupt rollout payload: {row['payload']}")
        return {"episodes": len(rows), "steps": sum(row["length"] for row in rows), "manifest": manifest}


__all__ = ["ROLLOUT_SCHEMA", "RolloutEpisode", "RolloutStore"]
