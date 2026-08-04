from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeOutcome:
    task_success: bool
    terminated: bool
    environment_truncated: bool
    local_time_limit: bool
    steps: int

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be nonnegative")

    @property
    def done(self) -> bool:
        return self.terminated or self.environment_truncated or self.local_time_limit

    def accumulate(self, *, success: bool, terminated: bool, truncated: bool, local_limit: bool) -> EpisodeOutcome:
        return EpisodeOutcome(
            task_success=self.task_success or success,
            terminated=self.terminated or terminated,
            environment_truncated=self.environment_truncated or truncated,
            local_time_limit=self.local_time_limit or local_limit,
            steps=self.steps + 1,
        )
