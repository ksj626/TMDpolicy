from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Stage(Protocol):
    def __call__(self, state: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class AlternatingRound:
    """Explicit round orchestration; individual stages remain replaceable."""

    evaluate: Stage
    collect_fresh_rollouts: Stage
    train_discriminator: Stage
    score_rollouts: Stage
    query_teacher: Stage
    update_student: Stage

    def run(self, state: dict[str, Any], round_index: int) -> dict[str, Any]:
        current = dict(state)
        current["round"] = round_index
        current = self.evaluate(current)
        current = self.collect_fresh_rollouts(current)
        current = self.train_discriminator(current)
        current = self.score_rollouts(current)
        current = self.query_teacher(current)
        # The stage contract requires both models to remain frozen during update.
        current["teacher_frozen"] = True
        current["discriminator_frozen"] = True
        current = self.update_student(current)
        current = self.evaluate(current)
        return current

