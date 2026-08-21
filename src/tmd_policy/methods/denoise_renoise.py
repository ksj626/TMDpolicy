"""Shared DMD2 denoise--renoise transitions for training and inference."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


Velocity = Callable[[Tensor, Tensor], Tensor]
TransitionCallback = Callable[[int, Tensor, Tensor], None]


def clean_prediction(value: Tensor, time: Tensor, velocity: Tensor) -> Tensor:
    if time.shape != (value.shape[0],) or velocity.shape != value.shape:
        raise ValueError("DMD2 clean prediction expects value/velocity [B,H,D] and time [B]")
    return value.float() - time.float()[:, None, None] * velocity.float()


def renoise(clean: Tensor, target_time: Tensor, noise: Tensor) -> Tensor:
    if clean.shape != noise.shape or target_time.numel() != 1:
        raise ValueError("DMD2 re-noising expects matching actions and one target time")
    target = target_time.float()
    return (1.0 - target) * clean.float() + target * noise.float()


def denoise_renoise_prefix(
    velocity: Velocity,
    noise: Tensor,
    time_grid: Tensor,
    prefix_steps: int,
    *,
    generator: torch.Generator | None = None,
    renoise_noises: Tensor | None = None,
    transition_callback: TransitionCallback | None = None,
) -> Tensor:
    """Return the noised state at ``prefix_steps`` from pure Gaussian noise."""

    total_steps = int(time_grid.numel() - 1)
    if time_grid.ndim != 1 or total_steps < 1 or not torch.all(time_grid[:-1] > time_grid[1:]):
        raise ValueError("time_grid must be strictly descending with at least one transition")
    if not 0 <= prefix_steps < total_steps:
        raise ValueError("prefix_steps must select a denoising step")
    if renoise_noises is not None and renoise_noises.shape != (prefix_steps, *noise.shape):
        raise ValueError("prefix re-noising noise has the wrong shape")
    value = noise.float()
    for index in range(prefix_steps):
        current = time_grid[index].expand(value.shape[0])
        clean = clean_prediction(value, current, velocity(value, current))
        fresh = (
            torch.randn(value.shape, device=value.device, dtype=torch.float32, generator=generator)
            if renoise_noises is None
            else renoise_noises[index].float()
        )
        next_value = renoise(clean, time_grid[index + 1], fresh)
        if transition_callback is not None:
            transition_callback(index, value, next_value)
        value = next_value
    return value


def denoise_renoise_sample(
    velocity: Velocity,
    noise: Tensor,
    time_grid: Tensor,
    *,
    generator: torch.Generator | None = None,
    renoise_noises: Tensor | None = None,
    step_callback: Callable[[], None] | None = None,
) -> Tensor:
    """Run all DMD2 steps and return the final clean prediction."""

    total_steps = int(time_grid.numel() - 1)
    if time_grid.ndim != 1 or total_steps < 1 or not torch.all(time_grid[:-1] > time_grid[1:]):
        raise ValueError("time_grid must be strictly descending with at least one transition")
    if renoise_noises is not None and renoise_noises.shape != (max(0, total_steps - 1), *noise.shape):
        raise ValueError("renoise_noises must be [num_steps-1,B,H,D]")
    value = noise.float()
    for index in range(total_steps):
        current = time_grid[index].expand(value.shape[0])
        clean = clean_prediction(value, current, velocity(value, current))
        if step_callback is not None:
            step_callback()
        if index + 1 == total_steps:
            return clean
        fresh = (
            torch.randn(value.shape, device=value.device, dtype=torch.float32, generator=generator)
            if renoise_noises is None
            else renoise_noises[index].float()
        )
        value = renoise(clean, time_grid[index + 1], fresh)
    raise AssertionError("unreachable")


__all__ = ["clean_prediction", "denoise_renoise_prefix", "denoise_renoise_sample", "renoise"]
