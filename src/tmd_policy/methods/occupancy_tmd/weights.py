from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class MismatchPrioritizationWeight:
    value: Tensor
    clipping_fraction: Tensor

    @classmethod
    def from_prefix_logits(
        cls, logits: Tensor, valid_mask: Tensor, *, minimum: float, maximum: float
    ) -> MismatchPrioritizationWeight:
        last = valid_mask.sum(dim=1) - 1
        final = logits[torch.arange(len(logits), device=logits.device), last]
        raw = torch.exp(-final.detach())
        clipped = raw.clamp(minimum, maximum)
        return cls(clipped, (clipped != raw).float().mean())


@dataclass(frozen=True)
class ImportanceRatio:
    value: Tensor
    effective_sample_size: Tensor

    @classmethod
    def from_log_probabilities(
        cls,
        *,
        current_log_probability: Tensor,
        behavior_log_probability: Tensor,
        maximum: float,
    ) -> ImportanceRatio:
        if current_log_probability.shape != behavior_log_probability.shape:
            raise ValueError("importance log probabilities must match")
        ratio = torch.exp(current_log_probability - behavior_log_probability).clamp(max=maximum)
        ess = ratio.sum().square() / ratio.square().sum().clamp_min(1e-12)
        return cls(ratio.detach(), ess.detach())
