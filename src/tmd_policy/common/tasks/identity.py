from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any


def normalize_instruction(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return " ".join(normalized.split())


def normalized_instruction_hash(value: str) -> str:
    return hashlib.sha256(normalize_instruction(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskIdentity:
    benchmark: str
    suite: str
    suite_task_index: int
    dataset_task_index: int
    dataset_episode_index: int
    canonical_task_uid: str
    instruction: str
    instruction_hash: str
    bddl_identifier: str
    bddl_file_hash: str | None
    environment_version: str
    source_dataset_id: str
    source_dataset_revision: str

    @classmethod
    def create(
        cls,
        *,
        benchmark: str,
        suite: str,
        suite_task_index: int,
        dataset_task_index: int,
        dataset_episode_index: int,
        instruction: str,
        bddl_identifier: str,
        bddl_file_hash: str | None,
        environment_version: str,
        source_dataset_id: str,
        source_dataset_revision: str,
    ) -> TaskIdentity:
        instruction_hash = normalized_instruction_hash(instruction)
        identity = {
            "benchmark": benchmark,
            "suite": suite,
            "suite_task_index": suite_task_index,
            "instruction_hash": instruction_hash,
            "bddl_identifier": bddl_identifier,
            "bddl_file_hash": bddl_file_hash,
        }
        uid = "task-" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        return cls(
            benchmark=benchmark,
            suite=suite,
            suite_task_index=suite_task_index,
            dataset_task_index=dataset_task_index,
            dataset_episode_index=dataset_episode_index,
            canonical_task_uid=uid,
            instruction=instruction,
            instruction_hash=instruction_hash,
            bddl_identifier=bddl_identifier,
            bddl_file_hash=bddl_file_hash,
            environment_version=environment_version,
            source_dataset_id=source_dataset_id,
            source_dataset_revision=source_dataset_revision,
        )

    def __post_init__(self) -> None:
        strings = (
            self.benchmark,
            self.suite,
            self.canonical_task_uid,
            self.instruction,
            self.bddl_identifier,
            self.environment_version,
            self.source_dataset_id,
            self.source_dataset_revision,
        )
        if any(not value.strip() for value in strings):
            raise ValueError("task identity strings must be nonempty")
        if min(self.suite_task_index, self.dataset_task_index, self.dataset_episode_index) < 0:
            raise ValueError("task and episode indices must be nonnegative")
        expected_instruction_hash = normalized_instruction_hash(self.instruction)
        if self.instruction_hash != expected_instruction_hash:
            raise ValueError("instruction hash does not match normalized instruction")
        if not re.fullmatch(r"[0-9a-f]{64}", self.instruction_hash):
            raise ValueError("instruction_hash must be SHA-256")
        if self.bddl_file_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", self.bddl_file_hash):
            raise ValueError("bddl_file_hash must be SHA-256 when present")
        identity = {
            "benchmark": self.benchmark,
            "suite": self.suite,
            "suite_task_index": self.suite_task_index,
            "instruction_hash": self.instruction_hash,
            "bddl_identifier": self.bddl_identifier,
            "bddl_file_hash": self.bddl_file_hash,
        }
        expected_uid = "task-" + hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:24]
        if self.canonical_task_uid != expected_uid:
            raise ValueError("canonical_task_uid does not match the task identity fields")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TaskIdentity:
        return cls(**value)

    def assert_same_task(self, other: TaskIdentity) -> None:
        if self.canonical_task_uid != other.canonical_task_uid:
            raise ValueError(
                "canonical task mismatch: "
                f"{self.canonical_task_uid} ({self.instruction!r}) != "
                f"{other.canonical_task_uid} ({other.instruction!r})"
            )
