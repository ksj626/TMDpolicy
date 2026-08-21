"""Numerically safe first-order GAN gradient capture."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn

from tmd_policy.backends.action_coordinates import safe_device_transfer


class _CapturedFirstOrderGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: Tensor, gradient: Tensor, reported_loss: Tensor) -> Tensor:
        if value.shape != gradient.shape or value.device != gradient.device:
            raise ValueError("captured first-order gradient must match its input tensor")
        ctx.save_for_backward(gradient)
        return reported_loss.detach().clone()

    @staticmethod
    def backward(ctx: Any, output_gradient: Tensor) -> tuple[Tensor, None, None]:
        (gradient,) = ctx.saved_tensors
        multiplier = safe_device_transfer(output_gradient, gradient.device).to(gradient.dtype)
        return multiplier * gradient, None, None


def stable_first_order_surrogate(
    loss: Tensor,
    value: Tensor,
    *,
    scales: tuple[float, ...] = (1.0, 2.0**-8, 2.0**-16, 2.0**-24),
) -> tuple[Tensor, float, Tensor]:
    if loss.numel() != 1 or not torch.isfinite(loss.detach()).all():
        raise RuntimeError("generator GAN loss contains NaN/Inf before backward")
    saw_connected = False
    saw_finite = False
    for scale in scales:
        raw = torch.autograd.grad(loss * scale, value, retain_graph=True, allow_unused=True)[0]
        if raw is None:
            continue
        saw_connected = True
        gradient = raw.float() / scale
        if not torch.isfinite(gradient).all():
            continue
        saw_finite = True
        if torch.count_nonzero(gradient) == 0:
            continue
        gradient = gradient.to(value.dtype).detach()
        surrogate = _CapturedFirstOrderGradient.apply(value, gradient, loss.detach())
        if not torch.isfinite(surrogate.detach()).all():
            raise RuntimeError("captured generator GAN surrogate changed a finite loss to NaN/Inf")
        return surrogate, scale, gradient
    if not saw_connected:
        raise RuntimeError("generator GAN gradient is disconnected")
    if saw_finite:
        raise RuntimeError("generator GAN gradient is exactly zero")
    raise RuntimeError("generator GAN gradient contains NaN/Inf at all safe backward scales")


@contextmanager
def frozen_parameters(module: nn.Module):
    states = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, state in zip(module.parameters(), states, strict=True):
            parameter.requires_grad_(state)


__all__ = ["frozen_parameters", "stable_first_order_surrogate"]
