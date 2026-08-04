from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ActionConvention:
    dimension: int = 7
    prediction_horizon: int = 50
    execution_horizon: int = 10
    minimum: float = -1.0
    maximum: float = 1.0


def validate_action_chunk(actions: Tensor, mask: Tensor, convention: ActionConvention) -> None:
    if actions.ndim != 3 or actions.shape[1:] != (convention.prediction_horizon, convention.dimension):
        raise ValueError("canonical action tensor must be [B,prediction_horizon,action_dim]")
    if mask.shape != actions.shape[:2] or mask.dtype != torch.bool or torch.any(mask.sum(1) == 0):
        raise ValueError("canonical action mask must be boolean, shape matched, and nonempty")
    if not torch.isfinite(actions).all() or torch.any(actions < convention.minimum) or torch.any(actions > convention.maximum):
        raise ValueError("canonical actions are nonfinite or out of range")
