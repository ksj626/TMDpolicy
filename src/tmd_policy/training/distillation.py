from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MismatchWeights:
    """Detached failure-emphasis weights; never an occupancy importance ratio."""

    values: Tensor


@dataclass(frozen=True)
class MismatchPrioritization:
    minimum: float = 0.5
    maximum: float = 2.0
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.minimum <= 0 or self.maximum < self.minimum or self.temperature <= 0:
            raise ValueError("invalid bounded weighting parameters")

    def from_expert_log_ratio(self, final_prefix_logit: Tensor) -> MismatchWeights:
        """Low q_E/q_pi increases supervision; output is bounded and detached."""
        mismatch = torch.sigmoid(-final_prefix_logit.detach() / self.temperature)
        return MismatchWeights(self.minimum + (self.maximum - self.minimum) * mismatch)


# Compatibility name for the prototype API. The returned value has the new,
# semantically explicit MismatchWeights type.
DistillationWeights = MismatchPrioritization


def combined_distillation_loss(
    expert_losses: Tensor,
    teacher_losses: Tensor,
    teacher_weights: MismatchWeights | Tensor | None = None,
    *,
    expert_coefficient: float = 1.0,
    teacher_coefficient: float = 1.0,
) -> Tensor:
    if expert_losses.ndim != 1 or teacher_losses.ndim != 1:
        raise ValueError("loss inputs must be per-sample vectors")
    if teacher_weights is None:
        teacher_term = teacher_losses.mean()
    else:
        values = teacher_weights.values if isinstance(teacher_weights, MismatchWeights) else teacher_weights
        weights = values.detach().to(teacher_losses)
        if weights.shape != teacher_losses.shape:
            raise ValueError("teacher_weights shape mismatch")
        teacher_term = (weights * teacher_losses).sum() / weights.sum().clamp_min(1e-8)
    return expert_coefficient * expert_losses.mean() + teacher_coefficient * teacher_term


def select_teacher_queries(
    final_prefix_logits: Tensor,
    budget: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Without-replacement selection biased toward low expert-likeness."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    budget = min(budget, final_prefix_logits.numel())
    if budget == 0:
        return torch.empty(0, dtype=torch.long, device=final_prefix_logits.device)
    probabilities = torch.softmax(-final_prefix_logits.detach(), dim=0)
    return torch.multinomial(probabilities, budget, replacement=False, generator=generator)
