from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .transition_head import InnerSourceMode, RecurrentTransitionHead, _source_mode


@dataclass
class MainBackboneOutput:
    transition: Tensor
    features: Tensor


class MainBackbone(Protocol):
    def __call__(self, context: Any, action_state: Tensor, outer_time: Tensor) -> MainBackboneOutput: ...


def gaussian_source_like(reference: Tensor, generator: torch.Generator | None = None) -> Tensor:
    """Draw a standard-normal source with the reference shape/device/dtype."""

    return torch.randn(
        reference.shape,
        device=reference.device,
        dtype=reference.dtype,
        generator=generator,
    )


def oracle_outer_integrate(actions: Tensor, noise: Tensor, outer_steps: int) -> Tensor:
    """Integrate the analytic velocity ``epsilon - A`` from t=1 to t=0."""
    if outer_steps < 1:
        raise ValueError("outer_steps must be positive")
    state = noise.clone()
    velocity = noise - actions
    dt = -1.0 / outer_steps
    for _ in range(outer_steps):
        state = state + dt * velocity
    return state


class TMDActionGenerator(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        transition_head: RecurrentTransitionHead,
        *,
        outer_steps: int = 2,
        inner_steps: int = 2,
        main_loss_weight: float = 0.0,
        transition_loss: str = "huber",
        inner_source_mode: InnerSourceMode | str = InnerSourceMode.GAUSSIAN_TM,
    ) -> None:
        super().__init__()
        if outer_steps < 1 or inner_steps < 1:
            raise ValueError("outer_steps and inner_steps must be positive")
        self.backbone = backbone
        self.transition_head = transition_head
        self.outer_steps = outer_steps
        self.inner_steps = inner_steps
        self.main_loss_weight = main_loss_weight
        self.transition_loss = transition_loss
        self.inner_source_mode = _source_mode(inner_source_mode)
        self.last_counts = {"main_backbone": 0, "transition_head": 0}

    def sample(
        self,
        context: Any,
        noise: Tensor,
        *,
        inner_noises: Tensor | None = None,
        inner_generator: torch.Generator | None = None,
    ) -> Tensor:
        """Sample actions with independently controllable outer and inner noise.

        ``noise`` is the outer source epsilon with shape ``[B,H,D]``.
        ``inner_noises``, when supplied, is ``[outer_steps,B,H,D]`` and contains
        one independent inner source Z per outer evaluation.
        """

        batch = noise.shape[0]
        expected_inner_shape = (self.outer_steps, *noise.shape)
        if inner_noises is not None and inner_noises.shape != expected_inner_shape:
            raise ValueError(
                f"inner_noises expected shape {expected_inner_shape}, got {inner_noises.shape}"
            )
        state = noise
        dt = -1.0 / self.outer_steps
        main_count = 0
        head_count = 0
        for index in range(self.outer_steps):
            time = torch.full((batch,), 1.0 + index * dt, device=noise.device, dtype=torch.float32)
            main = self.backbone(context, state, time)
            if inner_noises is None:
                inner_noise = gaussian_source_like(noise, inner_generator)
            else:
                inner_noise = inner_noises[index]
            transition = self.transition_head.refine(
                main.transition,
                state,
                time,
                main.features,
                inner_noise,
                inner_steps=self.inner_steps,
                mode=self.inner_source_mode,
            )
            state = state + dt * transition
            main_count += 1
            head_count += self.inner_steps
        self.last_counts = {"main_backbone": main_count, "transition_head": head_count}
        return state

    def transition_matching_loss(
        self,
        context: Any,
        actions: Tensor,
        valid_mask: Tensor | None = None,
        *,
        noise: Tensor | None = None,
        inner_noise: Tensor | None = None,
        outer_time: Tensor | None = None,
        reduction: str = "mean",
    ) -> dict[str, Tensor]:
        if noise is None:
            noise = torch.randn_like(actions)
        batch = actions.shape[0]
        if outer_time is None:
            outer_time = torch.rand(batch, device=actions.device).mul_(0.999).add_(0.001)
        if inner_noise is None:
            inner_noise = gaussian_source_like(actions)
        time_expanded = outer_time[:, None, None]
        outer_state = (1.0 - time_expanded) * actions + time_expanded * noise
        target_transition = noise - actions
        main = self.backbone(context, outer_state, outer_time)
        tm = self.transition_head.matching_loss(
            main.transition,
            target_transition,
            outer_state,
            outer_time,
            main.features,
            valid_mask,
            inner_noise=inner_noise,
            inner_steps=self.inner_steps,
            loss=self.transition_loss,
            reduction="none",
            mode=self.inner_source_mode,
        )
        main_errors = F.mse_loss(main.transition, target_transition, reduction="none")
        if valid_mask is not None:
            if valid_mask.shape != actions.shape[:2]:
                raise ValueError(f"valid mask expected {actions.shape[:2]}, got {valid_mask.shape}")
            mask = valid_mask.to(main_errors.dtype).unsqueeze(-1)
            denominator = valid_mask.sum(dim=1).to(main_errors.dtype) * main_errors.shape[-1]
            main_loss = (main_errors * mask).sum(dim=(1, 2)) / denominator.clamp_min(1)
        else:
            main_loss = main_errors.mean(dim=(1, 2))
        total = tm + self.main_loss_weight * main_loss
        losses = {"loss": total, "transition_matching": tm, "main_flow": main_loss}
        if reduction == "none":
            return losses
        if reduction != "mean":
            raise ValueError("reduction must be 'none' or 'mean'")
        return {name: value.mean() for name, value in losses.items()}
