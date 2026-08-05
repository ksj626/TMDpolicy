"""Stage-1 MeanFlow equations for action-transition matching."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn

from tmd_policy.methods.flow_objectives import shift_time, shifted_time_grid


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
    student_time_shift_gamma: float,
    meanflow_time_shift_gamma: float,
    discrete_outer_steps: int,
    discrete_inner_steps: int,
    generator: torch.Generator | None = None,
) -> MeanFlowBatch:
    """Sample TMD outer-grid times and the paper's continuous/discrete `(s,r)` pair."""

    if not 0 <= flow_matching_fraction <= 1:
        raise ValueError("flow_matching_fraction must be in [0,1]")
    batch = reference.shape[0]
    kwargs = {"device": reference.device, "dtype": reference.dtype, "generator": generator}
    outer_noise = torch.randn(reference.shape, **kwargs)
    inner_source = torch.randn(reference.shape, **kwargs)
    outer_grid = shifted_time_grid(
        discrete_outer_steps,
        student_time_shift_gamma,
        device=reference.device,
        dtype=reference.dtype,
    )
    inner_grid = shifted_time_grid(
        discrete_inner_steps,
        student_time_shift_gamma,
        device=reference.device,
        dtype=reference.dtype,
    )
    s = shift_time(
        torch.rand(batch, device=reference.device, dtype=reference.dtype, generator=generator),
        meanflow_time_shift_gamma,
    )
    # The non-FM target is the immediately preceding inference-grid point.
    preceding = torch.searchsorted(inner_grid, s.contiguous(), right=True) - 1
    r = inner_grid[preceding.clamp(min=0, max=discrete_inner_steps)]
    mixture = torch.rand(batch, device=reference.device, generator=generator) < flow_matching_fraction
    r = torch.where(mixture, s, r)
    outer_indices = torch.randint(
        1,
        discrete_outer_steps + 1,
        (batch,),
        device=reference.device,
        generator=generator,
    )
    outer_time = outer_grid[outer_indices]
    return MeanFlowBatch(outer_noise, inner_source, outer_time, s, r, mixture)


def meanflow_total_derivative(
    head: MeanFlowHead,
    y_s: Tensor,
    s: Tensor,
    r: Tensor,
    context: Tensor,
    conditional_velocity: Tensor,
    base_velocity: Tensor | None = None,
    inner_source: Tensor | None = None,
) -> Tensor:
    """Exact JVP `∂_s u + (∇_y u) v` with `r` and context held fixed."""

    def function(path: Tensor, time: Tensor) -> Tensor:
        residual = head(path, time, r, context)
        if base_velocity is None or inner_source is None:
            return residual
        return inner_source - (base_velocity + residual)

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
    base_velocity: Tensor | None = None,
    valid_coordinates: Tensor,
    normalization_constant_scale: float,
) -> dict[str, Tensor]:
    """MeanFlow target with stop-gradient and per-sample adaptive normalization."""

    if target_transition.shape != inner_source.shape or valid_coordinates.shape != target_transition.shape:
        raise ValueError("transition/source/valid-coordinate tensors must have identical shapes")
    if torch.any(valid_coordinates.flatten(1).sum(dim=1) == 0):
        raise ValueError("every TMD sample must contain a valid environment coordinate")
    if normalization_constant_scale <= 0:
        raise ValueError("normalization_constant_scale must be positive")
    y_s = (1.0 - _expand(inner_time, target_transition)) * target_transition + _expand(
        inner_time, target_transition
    ) * inner_source
    conditional_velocity = inner_source - target_transition
    residual = head(y_s, inner_time, target_time, context)
    if base_velocity is None:
        # Compatibility for explicitly adapted heads; faithful TM-MF always
        # supplies the frozen/current SmolVLA outer velocity.
        prediction = residual
    else:
        if base_velocity.shape != target_transition.shape:
            raise ValueError("base velocity must match the transition shape")
        prediction = inner_source - (base_velocity + residual)
    derivative = meanflow_total_derivative(
        head,
        y_s,
        inner_time,
        target_time,
        context,
        conditional_velocity,
        base_velocity=base_velocity,
        inner_source=inner_source if base_velocity is not None else None,
    )
    target = (
        conditional_velocity
        - _expand(inner_time - target_time, conditional_velocity) * derivative
    ).detach()
    squared = (prediction - target).square() * valid_coordinates.to(prediction.dtype)
    valid_count = valid_coordinates.flatten(1).sum(dim=1).to(prediction.dtype)
    squared_error_sum = squared.flatten(1).sum(dim=1)
    normalization_constant = normalization_constant_scale * valid_count
    adaptive = squared_error_sum / (squared_error_sum + normalization_constant).detach()
    return {
        "loss": adaptive.mean(),
        "per_sample_loss": adaptive,
        "squared_error_sum": squared_error_sum,
        "valid_coordinate_count": valid_count,
        "normalization_constant": normalization_constant,
        "prediction": prediction,
        "residual": residual,
        "base_velocity": base_velocity if base_velocity is not None else torch.zeros_like(prediction),
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
    student_time_shift_gamma: float,
    base_velocity: Tensor | None = None,
    step_callback: Callable[[], None] | None = None,
) -> Tensor:
    """Shared training/inference convention: descending Euler from `s=1` to `r=0`."""

    if num_steps < 1:
        raise ValueError("inner num_steps must be positive")
    grid = shifted_time_grid(
        num_steps,
        student_time_shift_gamma,
        device=source.device,
        dtype=source.dtype,
        descending=True,
    )
    value = source
    if base_velocity is not None and base_velocity.shape != source.shape:
        raise ValueError("inner-flow base velocity must match the source")
    batch = source.shape[0]
    for current, target in zip(grid[:-1], grid[1:], strict=True):
        s = current.expand(batch)
        r = target.expand(batch)
        residual = head(value, s, r, context)
        average_velocity = residual if base_velocity is None else source - (base_velocity + residual)
        value = value + (target - current) * average_velocity
        if step_callback is not None:
            step_callback()
    return value


__all__ = [
    "MeanFlowBatch",
    "integrate_inner_flow",
    "meanflow_loss",
    "meanflow_total_derivative",
    "sample_meanflow_batch",
]
