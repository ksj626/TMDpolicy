from __future__ import annotations

from pathlib import Path
from typing import Any

from .registry import TaskRegistry, TaskRegistryError


def inspect_cached_libero(cache_root: str | Path, registry_path: str | Path | None) -> dict[str, Any]:
    root = Path(cache_root)
    tasks_path = root / "meta" / "tasks.parquet"
    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not tasks_path.is_file() or not episode_files:
        return {"available": False, "cache_root": str(root), "reason": "cached metadata parquet missing"}
    import pandas as pd

    tasks = pd.read_parquet(tasks_path)
    episodes = pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)
    task_rows = [
        {"dataset_task_index": int(row["task_index"]), "instruction": str(instruction)}
        for instruction, row in tasks.iterrows()
    ]
    registry = TaskRegistry.from_json(registry_path) if registry_path and Path(registry_path).is_file() else None
    episode_rows, unresolved = [], []
    for _, row in episodes.iterrows():
        episode = int(row["episode_index"])
        mapped = None
        if registry is not None:
            try:
                first = registry.tasks[0]
                mapped = registry.by_dataset_episode(
                    first.source_dataset_id, first.source_dataset_revision, episode
                ).to_dict()
            except TaskRegistryError:
                pass
        if mapped is None:
            unresolved.append(episode)
        episode_rows.append({
            "episode_index": episode,
            "instructions": [str(value) for value in row["tasks"]],
            "canonical": mapped,
        })
    return {
        "available": True, "cache_root": str(root.resolve()), "dataset_tasks": task_rows,
        "episodes": episode_rows, "unresolved_episode_indices": unresolved,
        "ambiguous_or_unresolved": bool(unresolved),
    }
