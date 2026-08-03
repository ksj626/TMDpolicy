from __future__ import annotations

from pathlib import Path
from typing import Any

from tmd_policy.data.schemas import TeacherQuery
from tmd_policy.data.storage import ChunkStore


class TeacherQueryCache:
    def __init__(self, root: str | Path) -> None:
        self.store = ChunkStore(root)

    def get(
        self,
        observation_id: str,
        teacher_checkpoint: str,
        teacher_revision: str,
        processor_revision: str,
        sampling_seed: int,
        inference_steps: int,
        sample_index: int = 0,
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        key = TeacherQuery.make_cache_key(
            observation_id,
            teacher_checkpoint,
            teacher_revision,
            processor_revision,
            inference_steps,
            sampling_seed,
            sample_index,
        )
        for record in self.store.records(cache_key=key):
            return record, self.store.load_arrays(record)
        return None

    def put(self, query: TeacherQuery) -> None:
        if self.get(
            query.observation_id,
            query.teacher_checkpoint,
            query.teacher_revision,
            query.processor_revision,
            query.sampling_seed,
            query.inference_steps,
            query.sample_index,
        ) is not None:
            return
        self.store.append(query)
