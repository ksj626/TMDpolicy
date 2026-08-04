from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RectifiedFlowSchedule:
    minimum_time: float = 1e-3
    maximum_time: float = 0.999

    def __post_init__(self) -> None:
        if not 0 < self.minimum_time < self.maximum_time < 1:
            raise ValueError("score support must satisfy 0 < minimum_time < maximum_time < 1")

    def alpha(self, time: Tensor) -> Tensor:
        return 1 - time

    def sigma(self, time: Tensor) -> Tensor:
        return time

    def interpolate(self, data: Tensor, noise: Tensor, time: Tensor) -> Tensor:
        while time.ndim < data.ndim:
            time = time.unsqueeze(-1)
        return (1 - time) * data + time * noise

    def velocity_to_score(self, state: Tensor, velocity: Tensor, time: Tensor) -> Tensor:
        """Convert the conditional marginal velocity for the Cond-OT path.

        With `x_t=(1-t)x_0+t eps`, `eps=x_t+(1-t)v_t`; Tweedie's identity
        therefore gives `score=-(x_t+(1-t)v_t)/t`.
        """
        if state.shape != velocity.shape:
            raise ValueError("state and velocity must have identical shapes")
        if torch.any(time < self.minimum_time) or torch.any(time > self.maximum_time):
            raise ValueError("time lies outside the declared nonsingular score support")
        while time.ndim < velocity.ndim:
            time = time.unsqueeze(-1)
        return -(state + (1 - time) * velocity) / time
