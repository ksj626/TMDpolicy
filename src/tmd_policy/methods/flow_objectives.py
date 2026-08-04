"""Shared rectified-flow corruption, timestep, and DMD objective utilities."""

from __future__ import annotations

import torch
from torch import Tensor


def validate_time_distribution(
    minimum_time: float,
    maximum_time: float,
    time_shift_gamma: float,
) -> None:
    if not 0.0 <= minimum_time < maximum_time <= 1.0:
        raise ValueError("minimum_time and maximum_time must satisfy 0 <= min < max <= 1")
    if time_shift_gamma < 1.0:
        raise ValueError("time_shift_gamma must be at least 1")


def shift_time(unit_time: Tensor, gamma: float) -> Tensor:
    """Apply the TMD/DMD2-v rational timestep shift to values in ``[0, 1]``."""

    if gamma < 1.0:
        raise ValueError("time shift gamma must be at least 1")
    if torch.any((unit_time < 0) | (unit_time > 1)):
        raise ValueError("unshifted timestep must lie in [0,1]")
    return gamma * unit_time / ((gamma - 1.0) * unit_time + 1.0)


def sample_shifted_time(
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    minimum_time: float,
    maximum_time: float,
    gamma: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    validate_time_distribution(minimum_time, maximum_time, gamma)
    unit = torch.rand(batch_size, device=device, dtype=dtype, generator=generator)
    shifted = shift_time(unit, gamma)
    return minimum_time + (maximum_time - minimum_time) * shifted


def corrupt_rectified_flow(clean: Tensor, time: Tensor, noise: Tensor) -> Tensor:
    """Return ``a_tau=(1-tau)*a+tau*epsilon`` without detaching ``clean``."""

    if clean.shape != noise.shape:
        raise ValueError("clean action and corruption noise must have identical shapes")
    if time.shape != (clean.shape[0],):
        raise ValueError("corruption time must be [B]")
    weight = time
    while weight.ndim < clean.ndim:
        weight = weight.unsqueeze(-1)
    return (1.0 - weight) * clean + weight * noise


def executable_coordinate_mask(valid_timesteps: Tensor, width: int, *, executable_dim: int = 7) -> Tensor:
    if valid_timesteps.ndim != 2:
        raise ValueError("valid timestep mask must be [B,H]")
    if not 0 < executable_dim <= width:
        raise ValueError("executable_dim must be in [1,width]")
    dimensions = torch.arange(width, device=valid_timesteps.device) < executable_dim
    return valid_timesteps.bool().unsqueeze(-1) & dimensions[None, None]


def stopped_l1_score_direction(
    fake: Tensor,
    teacher: Tensor,
    valid_coordinates: Tensor,
    *,
    epsilon: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """TMD-v stopped fake-minus-teacher direction with per-sample valid L1 norm."""

    if fake.shape != teacher.shape or fake.shape != valid_coordinates.shape:
        raise ValueError("fake, teacher, and valid-coordinate tensors must have identical shapes")
    if epsilon <= 0:
        raise ValueError("normalization epsilon must be positive")
    mask = valid_coordinates.to(fake.dtype)
    count = mask.flatten(1).sum(dim=1)
    if torch.any(count == 0):
        raise ValueError("each sample needs at least one valid executable coordinate")
    difference = (fake - teacher) * mask
    numerator = difference.abs().flatten(1).sum(dim=1)
    denominator = numerator + epsilon
    direction = difference / denominator[:, None, None]
    direction = direction.detach()
    return direction, {
        "difference_l1": numerator.detach(),
        "denominator": denominator.detach(),
        "valid_coordinate_count": count.detach(),
        "teacher_abs_mean": (teacher.abs() * mask).flatten(1).sum(dim=1).div(count).detach(),
        "fake_abs_mean": (fake.abs() * mask).flatten(1).sum(dim=1).div(count).detach(),
        "direction_abs_mean": direction.abs().flatten(1).sum(dim=1).div(count).detach(),
    }


def stopped_dmd2_direction(
    fake: Tensor,
    teacher: Tensor,
    generated: Tensor,
    valid_coordinates: Tensor,
    *,
    epsilon: float,
) -> tuple[Tensor, dict[str, Tensor]]:
    """DMD2's fake-minus-real prediction direction and real-residual weight.

    The reference implementation forms ``p_real=x-pred_real`` and divides
    ``pred_fake-pred_real`` by the per-sample mean absolute ``p_real``. Masks
    restrict that image-space formula to executable action coordinates.
    """

    if (
        fake.shape != teacher.shape
        or fake.shape != generated.shape
        or fake.shape != valid_coordinates.shape
    ):
        raise ValueError("DMD2 prediction, generated, and mask tensors must have identical shapes")
    if epsilon <= 0:
        raise ValueError("normalization epsilon must be positive")
    mask = valid_coordinates.to(fake.dtype)
    count = mask.flatten(1).sum(dim=1)
    if torch.any(count == 0):
        raise ValueError("each DMD2 sample needs a valid executable coordinate")
    difference = (fake - teacher) * mask
    numerator = difference.abs().flatten(1).sum(dim=1)
    real_residual = ((generated - teacher).abs() * mask).flatten(1).sum(dim=1) / count
    denominator = real_residual + epsilon
    direction = (difference / denominator[:, None, None]).detach()
    return direction, {
        "difference_l1": numerator.detach(),
        "denominator": denominator.detach(),
        "valid_coordinate_count": count.detach(),
        "teacher_abs_mean": (teacher.abs() * mask).flatten(1).sum(dim=1).div(count).detach(),
        "fake_abs_mean": (fake.abs() * mask).flatten(1).sum(dim=1).div(count).detach(),
        "direction_abs_mean": direction.abs().flatten(1).sum(dim=1).div(count).detach(),
    }


def surrogate_vector_loss(generated: Tensor, direction: Tensor, valid_coordinates: Tensor) -> Tensor:
    """Construct a scalar whose gradient with respect to generated equals ``direction/N``."""

    if generated.shape != direction.shape or generated.shape != valid_coordinates.shape:
        raise ValueError("surrogate tensors must have identical shapes")
    mask = valid_coordinates.to(generated.dtype)
    return (generated * direction.detach() * mask).sum() / mask.sum().clamp_min(1.0)


__all__ = [
    "corrupt_rectified_flow",
    "executable_coordinate_mask",
    "sample_shifted_time",
    "shift_time",
    "stopped_dmd2_direction",
    "stopped_l1_score_direction",
    "surrogate_vector_loss",
    "validate_time_distribution",
]
