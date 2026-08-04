"""Stage-1 MeanFlow equations for action-transition matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn


def _expand(time: Tensor, reference: Tensor) -> Tensor:
    while time.ndim < reference.ndim:
        time = time.unsqueeze(-1)
    return time


class MeanFlowHead(Protocol):
    def __call__(self, y_s: Tensor, s: Tensor, r: Tensor, context: Tensor) -> Tensor: ...


@dataclass(frozen=True)
class MeanFlowBatch:
    outer_noise: Tensor
    inner_source: Tensor
    outer_time: Tensor
    inner_time: Tensor
    target_time: Tensor
    flow_matching_rows: Tensor


def sample_meanflow_batch(
    reference: Tensor,
    *,
    flow_matching_fraction: float,
    generator: torch.Generator | None = None,
) -> MeanFlowBatch:
    """Sample independent Gaussian sources and the paper's `0≤r≤s≤1` mixture."""

    if not 0 <= flow_matching_fraction <= 1:
        raise ValueError("flow_matching_fraction must be in [0,1]")
    batch = reference.shape[0]
    kwargs = {"device": reference.device, "dtype": reference.dtype, "generator": generator}
    outer_noise = torch.randn(reference.shape, **kwargs)
    inner_source = torch.randn(reference.shape, **kwargs)
    s = torch.rand(batch, device=reference.device, dtype=reference.dtype, generator=generator)
    r = torch.rand(batch, device=reference.device, dtype=reference.dtype, generator=generator) * s
    mixture = torch.rand(batch, device=reference.device, generator=generator) < flow_matching_fraction
    r = torch.where(mixture, s, r)
    outer_time = torch.rand(batch, device=reference.device, dtype=reference.dtype, generator=generator)
    return MeanFlowBatch(outer_noise, inner_source, outer_time, s, r, mixture)


def meanflow_total_derivative(
    head: MeanFlowHead,
    y_s: Tensor,
    s: Tensor,
    r: Tensor,
    context: Tensor,
    conditional_velocity: Tensor,
) -> Tensor:
    """Exact JVP `∂_s u + (∇_y u) v` with `r` and context held fixed."""

    def function(path: Tensor, time: Tensor) -> Tensor:
        return head(path, time, r, context)

    _, derivative = torch.func.jvp(
        function,
        (y_s, s),
        (conditional_velocity, torch.ones_like(s)),
    )
    return derivative


def meanflow_loss(
    head: MeanFlowHead,
    *,
    target_transition: Tensor,
    inner_source: Tensor,
    inner_time: Tensor,
    target_time: Tensor,
    context: Tensor,
    valid_coordinates: Tensor,
    normalization_constant: float,
) -> dict[str, Tensor]:
    """MeanFlow target with stop-gradient and per-sample adaptive normalization."""

    if target_transition.shape != inner_source.shape or valid_coordinates.shape != target_transition.shape:
        raise ValueError("transition/source/valid-coordinate tensors must have identical shapes")
    if torch.any(valid_coordinates.flatten(1).sum(dim=1) == 0):
        raise ValueError("every TMD sample must contain a valid environment coordinate")
    if normalization_constant <= 0:
        raise ValueError("normalization_constant must be positive")
    y_s = (1.0 - _expand(inner_time, target_transition)) * target_transition + _expand(
        inner_time, target_transition
    ) * inner_source
    conditional_velocity = inner_source - target_transition
    prediction = head(y_s, inner_time, target_time, context)
    derivative = meanflow_total_derivative(
        head, y_s, inner_time, target_time, context, conditional_velocity
    )
    target = (
        conditional_velocity
        - _expand(inner_time - target_time, conditional_velocity) * derivative
    ).detach()
    squared = (prediction - target).square() * valid_coordinates.to(prediction.dtype)
    denominator = valid_coordinates.flatten(1).sum(dim=1).to(prediction.dtype)
    per_sample_mse = squared.flatten(1).sum(dim=1) / denominator
    adaptive = per_sample_mse / (per_sample_mse.detach() + normalization_constant)
    return {
        "loss": adaptive.mean(),
        "per_sample_loss": adaptive,
        "prediction": prediction,
        "target": target,
        "inner_state": y_s,
        "conditional_velocity": conditional_velocity,
        "total_derivative": derivative,
    }


def integrate_inner_flow(
    head: MeanFlowHead,
    source: Tensor,
    context: Tensor,
    *,
    num_steps: int,
) -> Tensor:
    """Shared training/inference convention: descending Euler from `s=1` to `r=0`."""

    if num_steps < 1:
        raise ValueError("inner num_steps must be positive")
    grid = torch.linspace(1.0, 0.0, num_steps + 1, device=source.device, dtype=source.dtype)
    value = source
    batch = source.shape[0]
    for current, target in zip(grid[:-1], grid[1:], strict=True):
        s = current.expand(batch)
        r = target.expand(batch)
        value = value + (target - current) * head(value, s, r, context)
    return value


__all__ = [
    "MeanFlowBatch",
    "integrate_inner_flow",
    "meanflow_loss",
    "meanflow_total_derivative",
    "sample_meanflow_batch",
]
