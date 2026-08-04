"""Episode-disjoint, terminal-aware chunks from the real `lerobot/libero` dataset."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import Dataset

Split = Literal["train", "validation", "test"]
SCHEMA_NAME = "tmdpolicy.lerobot-libero-chunk/v1"


def _task_indices(metadata: Any, episode: int) -> list[int]:
    row = metadata.episodes[episode]
    for key in ("task_indices", "task_index"):
        if key in row:
            value = row[key]
            if hasattr(value, "tolist"):
                value = value.tolist()
            return [int(item) for item in value] if isinstance(value, (list, tuple)) else [int(value)]
    tasks = row.get("tasks", [])
    if isinstance(tasks, str):
        tasks = [tasks]
    return [int(metadata.get_task_index(task)) for task in tasks]


def _task_text(metadata: Any, task_index: int) -> str:
    tasks = metadata.tasks
    try:
        return str(tasks.iloc[task_index].name)
    except (AttributeError, KeyError, IndexError):
        try:
            return str(tasks[task_index])
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"unable to resolve LIBERO task {task_index} from dataset metadata") from error


def canonical_task_uid(task_index: int, instruction: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", instruction.lower()).strip("-")[:72]
    digest = hashlib.sha256(instruction.encode("utf-8")).hexdigest()[:8]
    return f"libero:{task_index:02d}:{slug}:{digest}"


def stratified_episode_split(
    episode_to_task: dict[int, int],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    """Split whole episodes within task; a frame can occur in exactly one split."""

    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1:
        raise ValueError("validation/test fractions must be in (0,1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation/test fractions must sum to less than one")
    grouped: dict[int, list[int]] = defaultdict(list)
    for episode, task in episode_to_task.items():
        grouped[task].append(episode)
    result = {"train": [], "validation": [], "test": []}
    for task, episodes in sorted(grouped.items()):
        rng = random.Random(seed + 1_000_003 * task)
        values = sorted(episodes)
        rng.shuffle(values)
        if len(values) < 3:
            raise ValueError(f"task {task} has {len(values)} episodes; three-way disjoint split is impossible")
        n_validation = max(1, round(len(values) * validation_fraction))
        n_test = max(1, round(len(values) * test_fraction))
        while n_validation + n_test >= len(values):
            if n_validation >= n_test and n_validation > 1:
                n_validation -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                raise ValueError(f"task {task} cannot retain a training episode")
        result["validation"].extend(values[:n_validation])
        result["test"].extend(values[n_validation : n_validation + n_test])
        result["train"].extend(values[n_validation + n_test :])
    for values in result.values():
        values.sort()
    assert_episode_disjoint(result)
    return result


def assert_episode_disjoint(splits: dict[str, list[int]]) -> None:
    names = ("train", "validation", "test")
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = set(splits[left]).intersection(splits[right])
            if overlap:
                raise ValueError(f"episode leakage between {left} and {right}: {sorted(overlap)[:10]}")


def build_episode_manifest(
    *,
    repo_id: str,
    revision: str,
    root: str | Path,
    output: str | Path,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    token: str | bool | None = None,
    expected_source_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve real Hub metadata and write the immutable episode split contract."""

    if repo_id != "lerobot/libero" or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("build-expert requires lerobot/libero at an immutable Hub revision")
    from collections import defaultdict
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from tmd_policy.backends.lerobot.compatibility import verify_installed_lerobot

    source = LeRobotDataset(
        repo_id,
        root=root,
        revision=revision,
        download_videos=False,
        token=token,
        video_backend="pyav",
    )

    if str(source.revision) != revision:
        raise RuntimeError(
            f"LeRobot resolved dataset revision {source.revision}, expected {revision}"
        )

    metadata = source.meta

    tasks_by_episode: dict[int, set[int]] = defaultdict(set)
    frame_metadata = source.reader.hf_dataset.select_columns(
        ["episode_index", "task_index"]
    )

    for row in frame_metadata:
        tasks_by_episode[int(row["episode_index"])].add(int(row["task_index"]))
    episode_to_task: dict[int, int] = {}
    for episode in range(int(metadata.total_episodes)):
        indices = sorted(tasks_by_episode.get(episode, set()))
        if len(indices) != 1:
            raise RuntimeError(f"LIBERO episode {episode} has ambiguous task identities: {indices}")
        episode_to_task[episode] = indices[0]
    splits = stratified_episode_split(
        episode_to_task,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )
    task_indices = sorted(set(episode_to_task.values()))
    task_table = {
        str(index): {
            "instruction": _task_text(metadata, index),
            "canonical_task_uid": canonical_task_uid(index, _task_text(metadata, index)),
        }
        for index in task_indices
    }
    manifest = {
        "schema": SCHEMA_NAME,
        "dataset_id": repo_id,
        "dataset_revision": revision,
        "fps": float(metadata.fps),
        "prediction_horizon": 50,
        "terminal_padding": "repeat boundary action; action_is_pad=True",
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "splits": splits,
        "episode_to_task": {str(key): value for key, value in sorted(episode_to_task.items())},
        "tasks": task_table,
        "lerobot": verify_installed_lerobot(expected_source_hashes=expected_source_hashes),
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_episode_manifest(path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA_NAME:
        raise ValueError(f"unsupported expert manifest schema: {manifest.get('schema')!r}")
    assert_episode_disjoint(manifest["splits"])
    if manifest.get("prediction_horizon") != 50:
        raise ValueError("expert manifest horizon must be 50")
    return manifest


class LeRobotLiberoChunks(Dataset):
    """Direct view of one manifest split with 50 future actions per frame."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: Split,
        *,
        root: str | Path,
        download_videos: bool = True,
        video_backend: str | None = None,
        token: str | bool | None = None,
    ) -> None:
        super().__init__()
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"unknown split: {split}")
        self.manifest = load_episode_manifest(manifest_path)
        self.split = split
        self.episodes = [int(value) for value in self.manifest["splits"][split]]
        fps = float(self.manifest["fps"])
        delta = {"action": [offset / fps for offset in range(50)]}
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset(
            self.manifest["dataset_id"],
            root=root,
            episodes=self.episodes,
            delta_timestamps=delta,
            revision=self.manifest["dataset_revision"],
            download_videos=download_videos,
            video_backend=video_backend,
            token=token,
        )
        self.dataset_revision = str(self.dataset.revision)
        if self.dataset_revision != self.manifest["dataset_revision"]:
            raise RuntimeError(
                f"LeRobot resolved dataset revision {self.dataset_revision}, expected {self.manifest['dataset_revision']}"
            )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        action = torch.as_tensor(sample["action"], dtype=torch.float32)
        is_pad = torch.as_tensor(sample["action_is_pad"], dtype=torch.bool)
        if action.shape != (50, 7) or is_pad.shape != (50,):
            raise RuntimeError(f"LeRobot returned invalid chunk shapes: action={action.shape}, mask={is_pad.shape}")
        episode = int(torch.as_tensor(sample["episode_index"]).item())
        if episode not in self.episodes:
            raise RuntimeError(f"reader leaked episode {episode} outside split {self.split}")
        task_index = int(torch.as_tensor(sample["task_index"]).item())
        instruction = str(sample["task"])
        expected = self.manifest["tasks"].get(str(task_index))
        if expected is None or expected["instruction"] != instruction:
            raise RuntimeError(f"task identity mismatch for task_index={task_index}")
        output = dict(sample)
        output["action"] = action
        output["action_is_pad"] = is_pad
        output["action_valid"] = ~is_pad
        output["canonical_task_uid"] = expected["canonical_task_uid"]
        output["task"] = instruction
        return output


__all__ = [
    "LeRobotLiberoChunks",
    "SCHEMA_NAME",
    "assert_episode_disjoint",
    "build_episode_manifest",
    "canonical_task_uid",
    "load_episode_manifest",
    "stratified_episode_split",
]
