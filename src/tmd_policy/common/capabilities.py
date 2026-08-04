from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class Capability(StrEnum):
    EXPERT_ACTION_CHUNKS = "expert_action_chunks"
    FLOW_VELOCITY = "flow_velocity"
    FLOW_SCORE = "flow_score"
    EXACT_LOG_DENSITY = "exact_log_density"
    STOCHASTIC_LOG_DENSITY = "stochastic_log_density"
    TOKEN_LOG_PROBABILITY = "token_log_probability"
    ON_POLICY_ROLLOUTS = "on_policy_rollouts"
    TEACHER_AT_STUDENT_ACTION = "teacher_at_student_action"
    PATH_WINDOWS = "path_windows"


class CapabilityError(RuntimeError):
    """Raised instead of silently substituting a different objective."""


@dataclass(frozen=True)
class CapabilitySet:
    values: frozenset[Capability]
    provider: str
    reason: str = ""

    @classmethod
    def of(cls, provider: str, *values: Capability, reason: str = "") -> CapabilitySet:
        return cls(frozenset(values), provider, reason)

    def require(self, required: Iterable[Capability], *, method: str) -> None:
        missing = sorted(set(required) - self.values)
        if missing:
            names = ", ".join(item.value for item in missing)
            detail = f" Provider report: {self.reason}" if self.reason else ""
            raise CapabilityError(f"{method} requires unavailable capabilities: {names}.{detail}")
