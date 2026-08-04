from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TeacherCacheIdentity:
    observation_content_hash: str
    canonical_task_uid: str
    teacher_model_revision: str
    teacher_processor_revision: str
    inference_schedule: tuple[float, ...]
    query_seed: int
    sample_index: int
    evaluated_student_action_hash: str | None

    @property
    def key(self) -> str:
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
