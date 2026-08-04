from __future__ import annotations

from pathlib import Path
from typing import Any

from tmd_policy.common.tasks import TaskRegistry
from tmd_policy.data.expert import build_expert_chunks, load_lerobot_expert_dataset

from .records import ExpertResearchRecord
from .splits import split_episodes
from .store import ResearchStore


def materialize_libero_expert(
    *, config: dict[str, Any], registry: TaskRegistry, output: str | Path,
    max_chunks: int | None = None, download_videos: bool = True,
) -> dict[str, Any]:
    dataset_config, task_config = config["dataset"], config["tasks"]
    rows = [task for task in registry.tasks if task.suite == task_config["suite"]]
    if not rows:
        raise ValueError(f"registry has no episodes for suite {task_config['suite']}")
    episode_to_uid = {task.dataset_episode_index: task.canonical_task_uid for task in rows}
    assignments = split_episodes(
        episode_to_uid,
        validation_fraction=float(dataset_config.get("validation_fraction", 0.1)),
        test_fraction=float(dataset_config.get("test_fraction", 0.1)),
        seed=int(config["training"]["seed"]),
    )
    cache = Path(config["dataset"]["cache"])
    dataset = load_lerobot_expert_dataset(
        dataset_config["id"], dataset_config["revision"], cache,
        sorted(episode_to_uid), 50, 10, download_videos=download_videos,
    )
    store = ResearchStore(output)
    count = 0
    for chunk in build_expert_chunks(
        dataset, dataset_id=dataset_config["id"], dataset_revision=dataset_config["revision"],
        prediction_horizon=50, execution_horizon=10, stride=int(dataset_config.get("stride", 10)),
        max_chunks=max_chunks, episode_splits=assignments,
    ):
        task = registry.by_dataset_episode(dataset_config["id"], dataset_config["revision"], chunk.episode_index)
        if task.dataset_task_index != chunk.task_index or task.instruction != chunk.instruction:
            raise ValueError(
                f"dataset/registry identity mismatch at episode {chunk.episode_index}: "
                f"{chunk.task_index}/{chunk.instruction!r} != "
                f"{task.dataset_task_index}/{task.instruction!r}"
            )
        arrays = {
            "action_plan_original": chunk.plan_actions.copy(),
            "action_plan_canonical": chunk.plan_actions.copy(),
            "action_valid": chunk.plan_valid,
            "state_sequence": chunk.path_states,
            "executed_path_actions": chunk.path_actions,
            "path_valid": chunk.path_valid,
            **{f"image::{key}": value for key, value in chunk.images.items()},
        }
        store.append(ExpertResearchRecord(
            kind="expert", sample_id=chunk.sample_id, task=task,
            episode_index=chunk.episode_index, frame_index=chunk.start_frame,
            split=chunk.split, arrays=arrays,
            metadata_extra={
                "observation_id": chunk.observation_id,
                "processor_revision": config["models"]["revisions"]["student_processor"],
                "normalizer_revision": config["models"]["revisions"]["student_processor"],
                "is_episode_start": chunk.is_episode_start,
                "reaches_episode_end": chunk.reaches_episode_end,
            },
        ))
        count += 1
    return {
        "records": count,
        "episodes": len(episode_to_uid),
        "splits": {name: sum(value == name for value in assignments.values()) for name in ("train", "validation", "test")},
        "store_audit": store.audit(),
    }
