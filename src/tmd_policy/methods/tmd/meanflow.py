from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import torch
from torch import Tensor, nn


def _broadcast(value: Tensor, reference: Tensor) -> Tensor:
    while value.ndim < reference.ndim:
        value = value.unsqueeze(-1)
    return value


@dataclass(frozen=True)
class MeanFlowConfig:
    action_dim: int = 7
    feature_dim: int = 64
    hidden_dim: int = 128
    finite_difference_delta: float = 0.005
    flow_matching_fraction: float = 0.75
    normalization_constant: float = 350.0
    derivative_mode: Literal["jvp", "finite_difference"] = "finite_difference"

    def __post_init__(self) -> None:
        if min(self.action_dim, self.feature_dim, self.hidden_dim) < 1:
            raise ValueError("MeanFlow dimensions must be positive")
        if self.finite_difference_delta <= 0 or not 0 <= self.flow_matching_fraction <= 1:
            raise ValueError("invalid finite-difference or r=s fraction")
        if self.normalization_constant <= 0 or self.derivative_mode not in {"jvp", "finite_difference"}:
            raise ValueError("invalid MeanFlow normalization/derivative mode")


class ActionMeanFlowHead(nn.Module):
    """Action head that cannot observe the independent inner source `y1`."""

    def __init__(self, config: MeanFlowConfig) -> None:
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.action_dim + config.feature_dim + 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(self, y_s: Tensor, s: Tensor, r: Tensor, features: Tensor) -> Tensor:
        if y_s.ndim != 3 or features.shape[:2] != y_s.shape[:2]:
            raise ValueError("y_s and features must be [B,H,D/F]")
        times = torch.stack((s, r), dim=-1)[:, None].expand(-1, y_s.shape[1], -1)
        return self.network(torch.cat((y_s, features, times), dim=-1))


def average_velocity(
    head: ActionMeanFlowHead,
    y_s: Tensor,
    s: Tensor,
    r: Tensor,
    features: Tensor,
    inner_source: Tensor,
) -> Tensor:
    # inner_source participates in the prescribed output preconditioning only;
    # ActionMeanFlowHead.forward never receives it.
    return inner_source - head(y_s, s, r, features)


def meanflow_total_derivative(
    head: ActionMeanFlowHead,
    y_s: Tensor,
    s: Tensor,
    r: Tensor,
    features: Tensor,
    inner_source: Tensor,
    conditional_velocity: Tensor,
    *,
    mode: str,
    delta: float,
) -> Tensor:
    if mode == "jvp":
        def function(path: Tensor, time: Tensor) -> Tensor:
            return average_velocity(head, path, time, r, features, inner_source)

        _, derivative = torch.func.jvp(
            function,
            (y_s, s),
            (conditional_velocity, torch.ones_like(s)),
        )
        return derivative
    if mode != "finite_difference":
        raise ValueError(f"unknown derivative mode: {mode}")
    upper_distance = torch.minimum(torch.full_like(s, delta), 1 - s)
    lower_distance = torch.minimum(torch.full_like(s, delta), s)
    upper = average_velocity(
        head,
        y_s + _broadcast(upper_distance, y_s) * conditional_velocity,
        s + upper_distance,
        r,
        features,
        inner_source,
    )
    lower = average_velocity(
        head,
        y_s - _broadcast(lower_distance, y_s) * conditional_velocity,
        s - lower_distance,
        r,
        features,
        inner_source,
    )
    denominator = _broadcast(upper_distance + lower_distance, y_s)
    if torch.any(denominator == 0):
        raise ValueError("finite difference has zero support at a boundary")
    return (upper - lower) / denominator


def meanflow_loss(
    head: ActionMeanFlowHead,
    *,
    outer_data: Tensor,
    outer_source: Tensor,
    outer_time: Tensor,
    inner_source: Tensor,
    inner_time: Tensor,
    target_time: Tensor,
    features: Tensor,
    valid_mask: Tensor,
    config: MeanFlowConfig,
) -> dict[str, Tensor]:
    if outer_data.shape != outer_source.shape or outer_data.shape != inner_source.shape:
        raise ValueError("outer data/source and inner source shapes must match")
    if valid_mask.shape != outer_data.shape[:2] or torch.any(valid_mask.sum(dim=1) == 0):
        raise ValueError("each MeanFlow sample needs a valid action")
    outer_state = (1 - _broadcast(outer_time, outer_data)) * outer_data + _broadcast(
        outer_time, outer_data
    ) * outer_source
    transition = outer_source - outer_data
    y_s = (1 - _broadcast(inner_time, transition)) * transition + _broadcast(
        inner_time, transition
    ) * inner_source
    conditional_velocity = inner_source - transition
    prediction = average_velocity(
        head, y_s, inner_time, target_time, features, inner_source
    )
    derivative = meanflow_total_derivative(
        head,
        y_s,
        inner_time,
        target_time,
        features,
        inner_source,
        conditional_velocity,
        mode=config.derivative_mode,
        delta=config.finite_difference_delta,
    )
    target = (
        conditional_velocity
        - _broadcast(inner_time - target_time, derivative) * derivative
    ).detach()
    squared = (prediction - target).square() * valid_mask.unsqueeze(-1)
    denominator = valid_mask.sum(dim=1) * outer_data.shape[-1]
    per_sample_squared = squared.sum(dim=(1, 2)) / denominator
    loss = per_sample_squared / (per_sample_squared.detach() + config.normalization_constant)
    return {
        "loss": loss.mean(),
        "per_sample_loss": loss,
        "outer_state": outer_state,
        "inner_state": y_s,
        "conditional_velocity": conditional_velocity,
        "average_velocity": prediction,
        "target": target,
        "total_derivative": derivative,
    }


def inner_flow_rollout(
    head: ActionMeanFlowHead,
    *,
    inner_source: Tensor,
    features: Tensor,
    time_grid: Tensor,
) -> Tensor:
    """Map noise to transition using the mathematically consistent Eq. 13 port.

    TMD Eq. 13 prints a plus sign although Eqs. 5 and 8 imply subtraction for
    a more-noisy-to-less-noisy map. The port uses `y_r=y_s-(s-r)u`; the contract
    records this action-flow adaptation until official code is released.
    """

    if time_grid.ndim != 1 or len(time_grid) < 2 or not torch.all(time_grid[:-1] > time_grid[1:]):
        raise ValueError("inner time grid must be strictly descending")
    value = inner_source
    batch = value.shape[0]
    for current, target in pairwise(time_grid):
        s = torch.full((batch,), float(current), device=value.device, dtype=value.dtype)
        r = torch.full((batch,), float(target), device=value.device, dtype=value.dtype)
        velocity = average_velocity(head, value, s, r, features, inner_source)
        value = value - _broadcast(s - r, value) * velocity
    return value
