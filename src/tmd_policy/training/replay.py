from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor


@dataclass(frozen=True)
class ImportanceWeights:
    """Detached density-ratio correction for historical replay only."""

    values: Tensor


@dataclass
class ReplayPools:
    """Keep exact-current, historical, and immutable teacher records disjoint."""

    fresh_current: list[dict[str, Any]]
    historical_replay: list[dict[str, Any]]
    teacher_query_cache: list[dict[str, Any]]

    def require_fresh_current(self, policy_version: str, minimum: int = 1) -> None:
        matching = [
            record for record in self.fresh_current if record.get("policy_version") == policy_version
        ]
        if len(matching) < minimum:
            raise RuntimeError(
                f"need {minimum} fresh_current records from exact policy {policy_version}, "
                f"found {len(matching)}; historical replay is not a substitute"
            )


@dataclass(frozen=True)
class ReplayRatioCorrection:
    logit_clip: float = 5.0
    weight_clip: float = 20.0

    def combine(self, expert_vs_replay: Tensor, replay_vs_current: Tensor) -> Tensor:
        if expert_vs_replay.shape != replay_vs_current.shape:
            raise ValueError("prefix logit shapes must match")
        return (expert_vs_replay + replay_vs_current).clamp(-self.logit_clip, self.logit_clip)

    def importance_weights(self, combined_logit: Tensor) -> ImportanceWeights:
        return ImportanceWeights(combined_logit.exp().clamp(max=self.weight_clip).detach())

    def weights(self, combined_logit: Tensor) -> Tensor:
        """Prototype compatibility API; prefer ``importance_weights(...).values``."""

        return self.importance_weights(combined_logit).values

    @staticmethod
    def effective_sample_size(weights: ImportanceWeights | Tensor) -> Tensor:
        values = weights.values if isinstance(weights, ImportanceWeights) else weights
        flat = values.reshape(-1).float()
        return flat.sum().square() / flat.square().sum().clamp_min(1e-12)
