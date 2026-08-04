from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .identity import TaskIdentity


class TaskRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class TaskRegistry:
    tasks: tuple[TaskIdentity, ...]

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("task registry cannot be empty")
        for key_name, key in (
            (
                "dataset episode",
                lambda item: (
                    item.source_dataset_id,
                    item.source_dataset_revision,
                    item.dataset_episode_index,
                ),
            ),
        ):
            values = [key(item) for item in self.tasks]
            if len(values) != len(set(values)):
                raise ValueError(f"duplicate {key_name} in task registry")

    @classmethod
    def from_json(cls, path: str | Path) -> TaskRegistry:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = value["tasks"] if isinstance(value, dict) else value
        return cls(tuple(TaskIdentity.from_dict(row) for row in rows))

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, "tasks": [item.to_dict() for item in self.tasks]}

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    def by_uid(self, uid: str) -> TaskIdentity:
        matches = [item for item in self.tasks if item.canonical_task_uid == uid]
        if not matches:
            raise TaskRegistryError(f"canonical task UID is not registered: {uid!r}")
        if any(
            (item.benchmark, item.suite, item.suite_task_index, item.instruction_hash, item.bddl_identifier)
            != (matches[0].benchmark, matches[0].suite, matches[0].suite_task_index,
                matches[0].instruction_hash, matches[0].bddl_identifier)
            for item in matches[1:]
        ):
            raise TaskRegistryError(f"canonical task UID has conflicting definitions: {uid!r}")
        return matches[0]

    def by_dataset_episode(self, dataset_id: str, revision: str, episode: int) -> TaskIdentity:
        matches = [
            item
            for item in self.tasks
            if (
                item.source_dataset_id,
                item.source_dataset_revision,
                item.dataset_episode_index,
            )
            == (dataset_id, revision, episode)
        ]
        if len(matches) != 1:
            raise TaskRegistryError(
                f"dataset episode has no unambiguous task mapping: {dataset_id}@{revision} episode={episode}"
            )
        return matches[0]

    def by_suite_task(self, benchmark: str, suite: str, suite_task_index: int) -> TaskIdentity:
        matches = [
            item
            for item in self.tasks
            if (item.benchmark, item.suite, item.suite_task_index)
            == (benchmark, suite, suite_task_index)
        ]
        if not matches or len({item.canonical_task_uid for item in matches}) != 1:
            raise TaskRegistryError(
                f"suite task has no unambiguous mapping: {benchmark}/{suite}/{suite_task_index}"
            )
        return matches[0]

    def assert_joinable(self, identities: Iterable[TaskIdentity]) -> str:
        values = tuple(identities)
        if not values:
            raise TaskRegistryError("cannot join an empty identity collection")
        uid = values[0].canonical_task_uid
        self.by_uid(uid)
        for item in values[1:]:
            values[0].assert_same_task(item)
        return uid
