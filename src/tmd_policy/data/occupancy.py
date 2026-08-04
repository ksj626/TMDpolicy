"""Real expert and student rollout windows for occupancy estimation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from tmd_policy.rollout.store import RolloutStore

from .libero import load_episode_manifest


def _visual_features(sample: dict[str, Any], length: int) -> Tensor:
    cameras = []
    for key in sorted(key for key in sample if key.startswith("observation.images.") and not key.endswith("_is_pad")):
        image = torch.as_tensor(sample[key], dtype=torch.float32)
        if image.ndim == 3:
            image = image.unsqueeze(0).expand(length, -1, -1, -1)
        if image.shape[0] != length:
            continue
        cameras.append(image.mean(dim=(-2, -1)))
        if len(cameras) == 2:
            break
    if not cameras:
        raise RuntimeError("occupancy windows require at least one real image sequence")
    while len(cameras) < 2:
        cameras.append(torch.zeros_like(cameras[0]))
    return torch.cat(cameras, dim=-1)


class ExpertOccupancyWindows(Dataset):
    """Episode-contiguous expert state/action/visual sequences."""

    source_label = 1.0

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        root: str | Path,
        window_length: int,
        download_videos: bool = True,
    ) -> None:
        self.manifest = load_episode_manifest(manifest_path)
        self.split = split
        self.window_length = window_length
        episodes = [int(value) for value in self.manifest["splits"][split]]
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        metadata = LeRobotDatasetMetadata(
            self.manifest["dataset_id"], root=root, revision=self.manifest["dataset_revision"]
        )
        delta_values = [offset / float(metadata.fps) for offset in range(window_length)]
        delta = {"action": delta_values, "observation.state": delta_values}
        delta.update({key: delta_values for key in metadata.video_keys})
        self.dataset = LeRobotDataset(
            self.manifest["dataset_id"],
            root=root,
            episodes=episodes,
            delta_timestamps=delta,
            revision=self.manifest["dataset_revision"],
            download_videos=download_videos,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def descriptor(self, index: int) -> tuple[int, int, int]:
        sample = self.dataset.reader.hf_dataset[index]
        task = int(torch.as_tensor(sample["task_index"]).item())
        frame = int(torch.as_tensor(sample["frame_index"]).item())
        return task, frame % self.window_length, 1

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        states = torch.as_tensor(sample["observation.state"], dtype=torch.float32)
        actions = torch.as_tensor(sample["action"], dtype=torch.float32)
        valid = ~torch.as_tensor(sample["action_is_pad"], dtype=torch.bool)
        state_pad = sample.get("observation.state_is_pad")
        if state_pad is not None:
            valid &= ~torch.as_tensor(state_pad, dtype=torch.bool)
        return {
            "state": states,
            "action": actions,
            "visual": _visual_features(sample, self.window_length),
            "valid": valid,
            "task_index": torch.as_tensor(sample["task_index"]).long(),
            "position": torch.arange(self.window_length),
            "source_label": torch.tensor(1.0),
            "episode_index": torch.as_tensor(sample["episode_index"]).long(),
        }


class StudentOccupancyWindows(Dataset):
    source_label = 0.0

    def __init__(self, store_path: str | Path, split: str, *, window_length: int, stride: int = 1) -> None:
        self.store = RolloutStore(store_path)
        self.window_length = window_length
        self.rows = [row for row in self.store.records() if row["split"] == split]
        self.windows: list[tuple[int, int]] = []
        for row_index, row in enumerate(self.rows):
            for start in range(0, row["length"] - window_length + 1, stride):
                self.windows.append((row_index, start))
        if not self.windows:
            raise ValueError(f"rollout store has no {split} windows of length {window_length}")

    def __len__(self) -> int:
        return len(self.windows)

    def descriptor(self, index: int) -> tuple[int, int, int]:
        row_index, start = self.windows[index]
        return int(self.rows[row_index]["task_index"]), start % self.window_length, 0

    def __getitem__(self, index: int) -> dict[str, Any]:
        row_index, start = self.windows[index]
        row = self.rows[row_index]
        payload = torch.load(self.store.root / row["payload"], map_location="cpu", weights_only=True)
        stop = start + self.window_length
        return {
            "state": payload["states"][start:stop].float(),
            "action": payload["actions"][start:stop].float(),
            "visual": payload["visual"][start:stop].float(),
            "valid": torch.ones(self.window_length, dtype=torch.bool),
            "task_index": torch.tensor(row["task_index"], dtype=torch.long),
            "position": torch.arange(self.window_length),
            "source_label": torch.tensor(0.0),
            "episode_index": torch.tensor(row["episode_id"], dtype=torch.long),
        }


class BalancedOccupancyDataset(Dataset):
    """Combine sources and attach inverse task/position/source frequency weights."""

    def __init__(self, expert: ExpertOccupancyWindows, student: StudentOccupancyWindows) -> None:
        self.datasets = (expert, student)
        self.lookup = [(source, index) for source, dataset in enumerate(self.datasets) for index in range(len(dataset))]
        descriptors = [self.datasets[source].descriptor(index) for source, index in self.lookup]
        self.counts = Counter(descriptors)
        inverse = torch.tensor([1.0 / self.counts[value] for value in descriptors], dtype=torch.float64)
        self.weights = (inverse / inverse.mean()).float()

    def __len__(self) -> int:
        return len(self.lookup)

    def __getitem__(self, index: int) -> dict[str, Any]:
        source, local = self.lookup[index]
        value = self.datasets[source][local]
        value["balance_weight"] = self.weights[index]
        return value


__all__ = ["BalancedOccupancyDataset", "ExpertOccupancyWindows", "StudentOccupancyWindows"]
