"""Expert and student replan-record units for short-window occupancy ratios."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from tmd_policy.rollout.store import RolloutStore

from .libero import load_episode_manifest


class ExpertOccupancyWindows(Dataset):
    """Expert chunk-start observation and its real canonical `[50,7]` plan."""

    source_label = 1

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        root: str | Path,
        window_length: int = 50,
        download_videos: bool = True,
        video_backend: str | None = None,
    ) -> None:
        if window_length != 50:
            raise ValueError("v2 occupancy records always discriminate full [50,7] plans")
        self.manifest = load_episode_manifest(manifest_path)
        self.split = split
        episodes = [int(value) for value in self.manifest["splits"][split]]
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        metadata = LeRobotDatasetMetadata(
            self.manifest["dataset_id"], root=root, revision=self.manifest["dataset_revision"]
        )
        delta = {"action": [offset / float(metadata.fps) for offset in range(50)]}
        self.dataset = LeRobotDataset(
            self.manifest["dataset_id"],
            root=root,
            episodes=episodes,
            delta_timestamps=delta,
            revision=self.manifest["dataset_revision"],
            download_videos=download_videos,
            video_backend=video_backend,
        )

    @property
    def task_support(self) -> set[int]:
        return {int(value) for value in self.manifest["episode_to_task"].values()}

    def __len__(self) -> int:
        return len(self.dataset)

    def descriptor(self, index: int) -> tuple[int, int, int]:
        sample = self.dataset.reader.hf_dataset[index]
        task = int(torch.as_tensor(sample["task_index"]).item())
        frame = int(torch.as_tensor(sample["frame_index"]).item())
        return task, frame // 50, self.source_label

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        action = torch.as_tensor(sample["action"], dtype=torch.float32)
        valid = ~torch.as_tensor(sample["action_is_pad"], dtype=torch.bool)
        if action.shape != (50, 7) or valid.shape != (50,):
            raise RuntimeError("expert occupancy sample violated the [50,7] plan contract")
        sample.update(
            {
                "action": action,
                "action_valid": valid,
                "executed_prefix_length": torch.tensor(int(valid.sum())),
                "source_label": torch.tensor(float(self.source_label)),
                "replan_position": torch.as_tensor(sample["frame_index"]).long(),
                "behavior_policy_checkpoint": "expert_dataset",
                "collection_round": torch.tensor(-1),
            }
        )
        return sample


class StudentOccupancyWindows(Dataset):
    """One item per actual student replan, never reconstructed from executed actions."""

    source_label = 0

    def __init__(
        self, store_path: str | Path, split: str, *, window_length: int = 50, stride: int = 1
    ) -> None:
        if window_length != 50 or stride != 1:
            raise ValueError("v2 student occupancy uses each stored full replan exactly once")
        self.store = RolloutStore(store_path)
        self.store.validate()
        self.rows = [row for row in self.store.records() if row["split"] == split]
        self.replans: list[tuple[int, int, dict[str, Any]]] = []
        for row_index, row in enumerate(self.rows):
            for replan_index, value in enumerate(self.store.load_replans(row)):
                self.replans.append((row_index, replan_index, value))
        if not self.replans:
            raise ValueError(f"rollout store has no {split} replan records")

    @property
    def task_support(self) -> set[int]:
        return {int(value[2]["global_task_index"]) for value in self.replans}

    def __len__(self) -> int:
        return len(self.replans)

    def descriptor(self, index: int) -> tuple[int, int, int]:
        _, _, value = self.replans[index]
        return int(value["global_task_index"]), int(value["environment_step"]) // 50, self.source_label

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, replan_index, value = self.replans[index]
        output: dict[str, Any] = {
            "observation.state": value["state"].float(),
            "action": value["planned_actions"].float(),
            "action_valid": torch.ones(50, dtype=torch.bool),
            "task_index": torch.tensor(value["global_task_index"], dtype=torch.long),
            "task": value["instruction"],
            "canonical_task_uid": value["canonical_task_uid"],
            "executed_prefix_length": torch.tensor(value["executed_prefix_length"]),
            "source_label": torch.tensor(float(self.source_label)),
            "replan_position": torch.tensor(value["environment_step"]),
            "episode_index": torch.tensor(self.rows[row_index]["episode_id"]),
            "behavior_policy_checkpoint": value["policy_checkpoint"],
            "collection_round": torch.tensor(value["collection_round"]),
            "replan_index": torch.tensor(replan_index),
        }
        for key, image in value["observations"].items():
            tensor = torch.as_tensor(image)
            output[key] = tensor[0] if tensor.ndim == 4 and tensor.shape[0] == 1 else tensor
        return output


class BalancedOccupancyDataset(Dataset):
    """Concatenate occupancy sources; balancing is performed by the train sampler."""

    def __init__(
        self,
        expert: ExpertOccupancyWindows,
        student: StudentOccupancyWindows,
        *,
        configured_tasks: Sequence[int] = tuple(range(40)),
    ) -> None:
        expected = set(int(value) for value in configured_tasks)
        if expected != set(range(40)):
            raise ValueError("occupancy configured task support must be all 40 canonical LIBERO tasks")
        if expert.task_support != expected or student.task_support != expected:
            raise ValueError(
                "occupancy task support mismatch: expert == student == configured == all 40 is required; "
                f"expert={sorted(expert.task_support)}, student={sorted(student.task_support)}"
            )
        self.datasets = (expert, student)
        self.lookup = [
            (source, index)
            for source, dataset in enumerate(self.datasets)
            for index in range(len(dataset))
        ]
        self.descriptors = [self.datasets[source].descriptor(index) for source, index in self.lookup]

    def __len__(self) -> int:
        return len(self.lookup)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source, local = self.lookup[index]
        return self.datasets[source][local]


class DeterministicStratifiedBatchSampler(Sampler[list[int]]):
    """Source-paired, task-stratified train batches with exact cursor resume."""

    def __init__(
        self,
        dataset: BalancedOccupancyDataset,
        batch_size: int,
        *,
        seed: int,
        epoch: int = 0,
        start_batch: int = 0,
        balance_position_bins: bool = False,
    ) -> None:
        if batch_size < 2 or batch_size % 2:
            raise ValueError("occupancy batch size must be positive and even")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.start_batch = start_batch
        self.balance_position_bins = balance_position_bins
        self.cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
        for index, (task, position, source) in enumerate(dataset.descriptors):
            bin_index = position if balance_position_bins else 0
            self.cells[(task, bin_index, source)].append(index)
        self.task_bins = sorted({(task, position) for task, position, _ in self.cells})
        missing = [
            (task, position, source)
            for task, position in self.task_bins
            for source in (0, 1)
            if (task, position, source) not in self.cells
        ]
        if missing:
            raise ValueError(f"occupancy sampler has absent task/source cells: {missing[:8]}")

    @property
    def batches_per_epoch(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __len__(self) -> int:
        return max(0, self.batches_per_epoch - self.start_batch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003)
        shuffled: dict[tuple[int, int, int], list[int]] = {}
        for cell, indices in sorted(self.cells.items()):
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled[cell] = [indices[index] for index in order]
        cursors = defaultdict(int)
        pair_count = self.batch_size // 2
        task_order = torch.randperm(len(self.task_bins), generator=generator).tolist()
        batches = []
        for batch_index in range(self.batches_per_epoch):
            batch = []
            for pair_index in range(pair_count):
                cell_key = self.task_bins[task_order[(batch_index * pair_count + pair_index) % len(task_order)]]
                for source in (1, 0):
                    cell = (*cell_key, source)
                    values = shuffled[cell]
                    cursor = cursors[cell]
                    batch.append(values[cursor % len(values)])
                    cursors[cell] += 1
            batches.append(batch)
        yield from batches[self.start_batch :]

    def state_dict(self) -> dict[str, int | bool]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "start_batch": self.start_batch,
            "balance_position_bins": self.balance_position_bins,
        }


__all__ = [
    "BalancedOccupancyDataset",
    "DeterministicStratifiedBatchSampler",
    "ExpertOccupancyWindows",
    "StudentOccupancyWindows",
]
