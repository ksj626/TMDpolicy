from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional

from tmd_policy.common.density import RectifiedFlowSchedule


def _reduce_masked(values: Tensor, valid_mask: Tensor) -> Tensor:
    if values.ndim != 3 or valid_mask.shape != values.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("values [B,H,D] and boolean mask [B,H] are required")
    if torch.any(valid_mask.sum(dim=1) == 0):
        raise ValueError("all-invalid DMD2 samples are forbidden")
    return (values * valid_mask.unsqueeze(-1)).sum(dim=(1, 2)) / (
        valid_mask.sum(dim=1) * values.shape[-1]
    )


def dmd2_distribution_matching_loss(
    generated: Tensor,
    *,
    time: Tensor,
    corruption_noise: Tensor,
    valid_mask: Tensor,
    real_velocity: Callable[[Tensor, Tensor], Tensor],
    fake_velocity: Callable[[Tensor, Tensor], Tensor],
    schedule: RectifiedFlowSchedule,
    weight: Tensor | None = None,
) -> dict[str, Tensor]:
    noisy = schedule.interpolate(generated, corruption_noise, time)
    with torch.no_grad():
        real_score = schedule.velocity_to_score(noisy, real_velocity(noisy, time), time)
        fake_score = schedule.velocity_to_score(noisy, fake_velocity(noisy, time), time)
        difference = real_score - fake_score
        normalizer = _reduce_masked(real_score.abs(), valid_mask).clamp_min(1e-8)
        gradient = -difference / normalizer[:, None, None]
        if weight is not None:
            if weight.shape != (generated.shape[0],):
                raise ValueError("DMD2 weights must be per-sample")
            gradient = gradient * weight[:, None, None]
        gradient = torch.nan_to_num(gradient)
    # Its derivative with respect to generated is exactly the detached DMD2
    # generator vector field, without differentiating either score network.
    per_sample = _reduce_masked(generated * gradient.detach(), valid_mask)
    return {
        "loss": per_sample.mean(),
        "per_sample_loss": per_sample,
        "generator_gradient": gradient,
        "real_score": real_score,
        "fake_score": fake_score,
    }


def fake_score_loss(
    prediction: Tensor,
    target_velocity: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    return _reduce_masked((prediction - target_velocity).square(), valid_mask).mean()


class ConditionalActionGAN(nn.Module):
    """DMD2 conditional GAN; deliberately unrelated to occupancy prefixes."""

    def __init__(self, action_dim: int, condition_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.condition = nn.Linear(condition_dim, hidden_dim)
        self.classifier = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, noisy_actions: Tensor, condition: Tensor, valid_mask: Tensor) -> Tensor:
        if noisy_actions.ndim != 3 or valid_mask.shape != noisy_actions.shape[:2]:
            raise ValueError("GAN actions/mask shape mismatch")
        encoded = self.action_encoder(noisy_actions)
        pooled = (encoded * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_mask.sum(dim=1, keepdim=True)
        return self.classifier(pooled + self.condition(condition)).squeeze(-1)


def discriminator_loss(real_logits: Tensor, fake_logits: Tensor) -> Tensor:
    return functional.softplus(-real_logits).mean() + functional.softplus(fake_logits).mean()


def generator_gan_loss(fake_logits: Tensor) -> Tensor:
    return functional.softplus(-fake_logits).mean()
