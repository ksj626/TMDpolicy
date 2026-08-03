from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import ExpertChunk


def episode_split(
    episode_to_task: dict[int, int], held_out_fraction: float = 0.1, seed: int = 17
) -> tuple[list[int], list[int]]:
    """Task-stratified split whose indivisible unit is a complete episode."""
    if not 0.0 <= held_out_fraction < 1.0:
        raise ValueError("held_out_fraction must be in [0, 1)")
    grouped: dict[int, list[int]] = defaultdict(list)
    for episode, task in episode_to_task.items():
        grouped[task].append(episode)
    rng = random.Random(seed)
    train: list[int] = []
    held_out: list[int] = []
    for episodes in grouped.values():
        rng.shuffle(episodes)
        count = round(len(episodes) * held_out_fraction)
        if held_out_fraction and len(episodes) > 1:
            count = max(1, min(count, len(episodes) - 1))
        held_out.extend(episodes[:count])
        train.extend(episodes[count:])
    return sorted(train), sorted(held_out)


def episode_split_three_way(
    episode_to_task: dict[int, int],
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 17,
) -> tuple[list[int], list[int], list[int]]:
    """Task-stratified train/validation/test split over indivisible episodes."""

    if (
        not 0 <= validation_fraction < 1
        or not 0 <= test_fraction < 1
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError("validation/test fractions must be nonnegative and sum to less than one")
    grouped: dict[int, list[int]] = defaultdict(list)
    for episode, task in episode_to_task.items():
        grouped[task].append(episode)
    rng = random.Random(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for episodes in grouped.values():
        rng.shuffle(episodes)
        count = len(episodes)
        validation_count = round(count * validation_fraction)
        test_count = round(count * test_fraction)
        if count >= 3 and validation_fraction:
            validation_count = max(1, validation_count)
        if count >= 3 and test_fraction:
            test_count = max(1, test_count)
        overflow = max(0, validation_count + test_count - (count - 1))
        while overflow and test_count:
            test_count -= 1
            overflow -= 1
        while overflow and validation_count:
            validation_count -= 1
            overflow -= 1
        validation.extend(episodes[:validation_count])
        test.extend(episodes[validation_count : validation_count + test_count])
        train.extend(episodes[validation_count + test_count :])
    return sorted(train), sorted(validation), sorted(test)


def make_observation_id(dataset_id: str, revision: str, episode: int, frame: int) -> str:
    raw = f"{dataset_id}@{revision}:episode={episode}:frame={frame}".encode()
    return hashlib.sha256(raw).hexdigest()[:24]


def build_expert_chunks(
    dataset: Any,
    *,
    dataset_id: str,
    dataset_revision: str,
    prediction_horizon: int = 50,
    execution_horizon: int = 10,
    stride: int = 10,
    max_chunks: int | None = None,
    episode_splits: dict[int, str] | None = None,
) -> Iterator[ExpertChunk]:
    """Build chunk-aligned records from a LeRobotDataset with delta windows."""
    emitted = 0
    for relative_index in range(len(dataset)):
        raw = dataset.get_raw_item(relative_index)
        frame = int(np.asarray(raw["frame_index"]).item())
        if frame % stride:
            continue
        item = dataset[relative_index]
        episode = int(item["episode_index"].item())
        task = int(item["task_index"].item())
        plan = item["action"].detach().cpu().numpy()
        states = item["observation.state"].detach().cpu().numpy()
        plan_pad = item.get("action_is_pad")
        state_pad = item.get("observation.state_is_pad")
        plan_valid = (
            np.ones(prediction_horizon, dtype=bool)
            if plan_pad is None
            else ~plan_pad.detach().cpu().numpy().astype(bool)
        )
        state_valid = (
            np.ones(execution_horizon + 1, dtype=bool)
            if state_pad is None
            else ~state_pad.detach().cpu().numpy().astype(bool)
        )
        path_valid = plan_valid[:execution_horizon] & state_valid[:-1] & state_valid[1:]
        images = {
            key: value.detach().cpu().numpy()
            for key, value in item.items()
            if key.startswith("observation.images.") and not key.endswith("_is_pad")
        }
        observation_id = make_observation_id(dataset_id, dataset_revision, episode, frame)
        sample_id = f"expert-{episode:05d}-{frame:04d}"
        yield ExpertChunk(
            sample_id=sample_id,
            observation_id=observation_id,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            episode_index=episode,
            task_index=task,
            instruction=str(item["task"]),
            start_frame=frame,
            plan_actions=plan[:prediction_horizon],
            plan_valid=plan_valid,
            path_states=states[: execution_horizon + 1],
            path_actions=plan[:execution_horizon],
            path_valid=path_valid,
            images=images,
            is_episode_start=frame == 0,
            reaches_episode_end=not bool(plan_valid[-1]),
            split=(episode_splits or {}).get(episode, "train"),
        )
        emitted += 1
        if max_chunks is not None and emitted >= max_chunks:
            return


def load_lerobot_expert_dataset(
    repo_id: str,
    revision: str,
    root: str | Path,
    episodes: Iterable[int],
    prediction_horizon: int = 50,
    execution_horizon: int = 10,
    *,
    download_videos: bool = True,
) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    fps = 10
    delta = {
        "action": [index / fps for index in range(prediction_horizon)],
        "observation.state": [index / fps for index in range(execution_horizon + 1)],
    }
    return LeRobotDataset(
        repo_id,
        root=root,
        episodes=list(episodes),
        delta_timestamps=delta,
        revision=revision,
        download_videos=download_videos,
        return_uint8=True,
    )
