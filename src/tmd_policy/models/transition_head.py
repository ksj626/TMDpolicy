from __future__ import annotations

import math
from enum import StrEnum

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class InnerSourceMode(StrEnum):
    """Named inner-flow constructions; modes never alias one another."""

    GAUSSIAN_TM = "gaussian_tm"
    ANCHORED_TM_ABLATION = "anchored_tm_ablation"
    GAUSSIAN_TM_MEANFLOW = "gaussian_tm_meanflow"


def _source_mode(value: InnerSourceMode | str) -> InnerSourceMode:
    try:
        mode = InnerSourceMode(value)
    except ValueError as error:
        choices = ", ".join(item.value for item in InnerSourceMode)
        raise ValueError(f"unknown inner source mode {value!r}; choose one of: {choices}") from error
    if mode is InnerSourceMode.GAUSSIAN_TM_MEANFLOW:
        raise NotImplementedError(
            "gaussian_tm_meanflow is reserved for a future mean-flow implementation"
        )
    return mode


def scalar_embedding(value: Tensor, dimension: int) -> Tensor:
    if dimension % 2:
        raise ValueError("embedding dimension must be even")
    value = value.float().reshape(-1, 1)
    frequencies = torch.exp(
        torch.linspace(0.0, -math.log(10_000.0), dimension // 2, device=value.device)
    )
    phase = value * frequencies.reshape(1, -1)
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class RecurrentTransitionHead(nn.Module):
    """Residual estimator for a Gaussian conditional flow over action velocities.

    ``backbone_transition`` is the reference ``B``. The network predicts
    ``Delta`` so that the estimated clean transition is ``B + Delta``. In the
    default Gaussian construction, ``inner_state`` follows the analytic path
    from clean transition ``Y`` at ``s=0`` to Gaussian source ``Z`` at ``s=1``.
    """

    def __init__(
        self,
        action_dim: int,
        backbone_feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        prediction_horizon: int = 50,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_horizon = prediction_horizon
        time_dim = min(64, hidden_dim if hidden_dim % 2 == 0 else hidden_dim - 1)
        self.time_dim = time_dim
        self.feature_projection = nn.Linear(backbone_feature_dim, hidden_dim)
        # inner state, Gaussian source, backbone reference, and outer state.
        self.input_projection = nn.Linear(action_dim * 4 + time_dim * 2, hidden_dim)
        self.position_embedding = nn.Embedding(prediction_horizon, hidden_dim)
        self.cells = nn.ModuleList([nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, action_dim)
        # Delta=0 is a deliberate safe initialization: Gaussian refinement then
        # returns the pretrained backbone reference exactly.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def gaussian_path(target_transition: Tensor, source_noise: Tensor, inner_time: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``Y_s=(1-s)Y+sZ`` and its exact derivative ``Z-Y``."""

        if target_transition.shape != source_noise.shape:
            raise ValueError("target transition and inner source must have identical shapes")
        if inner_time.ndim == 0:
            inner_time = inner_time.expand(target_transition.shape[0])
        if inner_time.shape != (target_transition.shape[0],):
            raise ValueError("inner_time must be scalar or [batch]")
        weight = inner_time.to(dtype=target_transition.dtype)[:, None, None]
        path = (1.0 - weight) * target_transition + weight * source_noise
        return path, source_noise - target_transition

    def initial_hidden(self, backbone_features: Tensor) -> list[Tensor]:
        if backbone_features.ndim != 3:
            raise ValueError("backbone features must be [batch, horizon, feature]")
        seed = torch.tanh(self.feature_projection(backbone_features))
        return [seed.clone() for _ in range(self.num_layers)]

    def step(
        self,
        inner_state: Tensor,
        source_noise: Tensor,
        backbone_transition: Tensor,
        outer_state: Tensor,
        outer_time: Tensor,
        inner_time: Tensor,
        backbone_features: Tensor,
        hidden: list[Tensor] | None = None,
    ) -> tuple[Tensor, list[Tensor]]:
        """Predict the residual ``Delta`` and advance recurrent hidden state."""

        batch, horizon, action_dim = inner_state.shape
        expected = inner_state.shape
        for name, value in {
            "source_noise": source_noise,
            "backbone_transition": backbone_transition,
            "outer_state": outer_state,
        }.items():
            if value.shape != expected:
                raise ValueError(f"{name} expected shape {expected}, got {value.shape}")
        if action_dim != self.action_dim or horizon > self.prediction_horizon:
            raise ValueError(
                f"head expected horizon <= {self.prediction_horizon}, action dim {self.action_dim}; "
                f"got {inner_state.shape}"
            )
        if hidden is None:
            hidden = self.initial_hidden(backbone_features)
        outer_emb = scalar_embedding(outer_time, self.time_dim)[:, None].expand(-1, horizon, -1)
        inner_emb = scalar_embedding(inner_time, self.time_dim)[:, None].expand(-1, horizon, -1)
        inputs = torch.cat(
            (
                inner_state,
                source_noise,
                backbone_transition,
                outer_state,
                outer_emb,
                inner_emb,
            ),
            dim=-1,
        )
        inputs = self.input_projection(inputs)
        positions = self.position_embedding(torch.arange(horizon, device=inputs.device))[None]
        layer_input = inputs + positions + self.feature_projection(backbone_features)
        next_hidden: list[Tensor] = []
        for cell, norm, previous in zip(self.cells, self.norms, hidden, strict=True):
            flat = cell(layer_input.reshape(batch * horizon, -1), previous.reshape(batch * horizon, -1))
            current = norm(flat.reshape(batch, horizon, self.hidden_dim))
            next_hidden.append(current)
            layer_input = self.dropout(current)
        return self.output_projection(layer_input), next_hidden

    def refine(
        self,
        backbone_transition: Tensor,
        outer_state: Tensor,
        outer_time: Tensor,
        backbone_features: Tensor,
        inner_noise: Tensor | None,
        *,
        inner_steps: int,
        mode: InnerSourceMode | str = InnerSourceMode.GAUSSIAN_TM,
    ) -> Tensor:
        """Integrate the selected inner construction from ``s=1`` to ``s=0``."""

        if inner_steps < 1:
            raise ValueError("inner_steps must be positive")
        source_mode = _source_mode(mode)
        hidden = self.initial_hidden(backbone_features)
        ds = -1.0 / inner_steps
        if source_mode is InnerSourceMode.GAUSSIAN_TM:
            if inner_noise is None or inner_noise.shape != backbone_transition.shape:
                raise ValueError("gaussian_tm requires shape-matched inner_noise")
            source = inner_noise
            current = source.clone()
        else:
            source = backbone_transition
            current = backbone_transition
        for index in range(inner_steps):
            inner_time = torch.full_like(outer_time, 1.0 + index * ds)
            residual, hidden = self.step(
                current,
                source,
                backbone_transition,
                outer_state,
                outer_time,
                inner_time,
                backbone_features,
                hidden,
            )
            if source_mode is InnerSourceMode.GAUSSIAN_TM:
                predicted_velocity = source - (backbone_transition + residual)
            else:
                predicted_velocity = residual
            current = current + ds * predicted_velocity
        return current

    def matching_loss(
        self,
        backbone_transition: Tensor,
        target_transition: Tensor,
        outer_state: Tensor,
        outer_time: Tensor,
        backbone_features: Tensor,
        valid_mask: Tensor | None = None,
        *,
        inner_noise: Tensor | None = None,
        inner_steps: int = 2,
        loss: str = "huber",
        reduction: str = "mean",
        mode: InnerSourceMode | str = InnerSourceMode.GAUSSIAN_TM,
    ) -> Tensor:
        """Compute transition matching, reducing valid coordinates per sample first."""

        if inner_steps < 1:
            raise ValueError("inner_steps must be positive")
        if reduction not in {"none", "mean"}:
            raise ValueError("reduction must be 'none' or 'mean'")
        source_mode = _source_mode(mode)
        if source_mode is InnerSourceMode.GAUSSIAN_TM:
            if inner_noise is None or inner_noise.shape != target_transition.shape:
                raise ValueError("gaussian_tm requires shape-matched inner_noise")
            source = inner_noise
            target_velocity = source - target_transition
        else:
            source = backbone_transition.detach()
            target_velocity = source - target_transition

        hidden = self.initial_hidden(backbone_features)
        per_step: list[Tensor] = []
        for index in range(inner_steps):
            inner_time_value = 1.0 - index / inner_steps
            inner_time = torch.full_like(outer_time, inner_time_value)
            analytic_state, _ = self.gaussian_path(target_transition, source, inner_time)
            residual, hidden = self.step(
                analytic_state,
                source,
                backbone_transition,
                outer_state,
                outer_time,
                inner_time,
                backbone_features,
                hidden,
            )
            if source_mode is InnerSourceMode.GAUSSIAN_TM:
                predicted_velocity = source - (backbone_transition + residual)
            else:
                predicted_velocity = residual
            if loss == "mse":
                error = (predicted_velocity - target_velocity).square()
            elif loss == "huber":
                error = F.smooth_l1_loss(predicted_velocity, target_velocity, reduction="none")
            else:
                raise ValueError(f"unknown transition loss: {loss}")
            per_step.append(error)

        errors = torch.stack(per_step, dim=0).mean(dim=0)
        if valid_mask is None:
            per_sample = errors.mean(dim=(1, 2))
        else:
            if valid_mask.shape != errors.shape[:2]:
                raise ValueError(f"valid mask expected {errors.shape[:2]}, got {valid_mask.shape}")
            mask = valid_mask.to(errors.dtype).unsqueeze(-1)
            denominator = valid_mask.sum(dim=1).to(errors.dtype) * errors.shape[-1]
            per_sample = (errors * mask).sum(dim=(1, 2)) / denominator.clamp_min(1)
        return per_sample if reduction == "none" else per_sample.mean()


__all__ = ["InnerSourceMode", "RecurrentTransitionHead", "scalar_embedding"]
