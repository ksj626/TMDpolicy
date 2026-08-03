from __future__ import annotations

from enum import StrEnum

import torch
from torch import Tensor, nn


class DiscriminatorVariant(StrEnum):
    POINTWISE = "pointwise"
    FINAL = "final"
    PREFIX = "prefix"


class PathNormalizer(nn.Module):
    """Train-split-only statistics for canonical state/action paths."""

    def __init__(self, state_dim: int = 8, action_dim: int = 7, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.epsilon = epsilon
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_scale", torch.ones(state_dim))
        self.register_buffer("action_mean", torch.zeros(action_dim))
        self.register_buffer("action_scale", torch.ones(action_dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(
        self,
        states: Tensor,
        actions: Tensor,
        valid_mask: Tensor,
        *,
        split: str,
    ) -> None:
        if split != "train":
            raise ValueError("PathNormalizer may only be fit on the training split")
        if states.shape[:2] != (actions.shape[0], actions.shape[1] + 1):
            raise ValueError("states must contain one more time point than actions")
        if valid_mask.shape != actions.shape[:2] or not valid_mask.any():
            raise ValueError("valid_mask must select at least one training transition")
        transition_states = torch.cat((states[:, :-1], states[:, 1:]), dim=1)
        state_mask = torch.cat((valid_mask, valid_mask), dim=1)
        selected_states = transition_states[state_mask]
        selected_actions = actions[valid_mask]
        self.state_mean.copy_(selected_states.mean(dim=0))
        self.state_scale.copy_(selected_states.std(dim=0, unbiased=False).clamp_min(self.epsilon))
        self.action_mean.copy_(selected_actions.mean(dim=0))
        self.action_scale.copy_(selected_actions.std(dim=0, unbiased=False).clamp_min(self.epsilon))
        self.fitted.fill_(True)

    def forward(self, states: Tensor, actions: Tensor) -> tuple[Tensor, Tensor]:
        if not bool(self.fitted):
            raise RuntimeError("PathNormalizer must be fit on the train split before use")
        return (
            (states - self.state_mean) / self.state_scale,
            (actions - self.action_mean) / self.action_scale,
        )


class CausalPathDiscriminator(nn.Module):
    """Low-dimensional causal discriminator with one token per transition."""

    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 7,
        execution_horizon: int = 10,
        num_tasks: int = 40,
        model_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        variant: DiscriminatorVariant | str = DiscriminatorVariant.PREFIX,
        normalizer: PathNormalizer | None = None,
    ) -> None:
        super().__init__()
        self.execution_horizon = execution_horizon
        self.variant = DiscriminatorVariant(variant)
        self.normalizer = normalizer or PathNormalizer(state_dim, action_dim)
        raw_dim = state_dim * 3 + action_dim
        self.transition_projection = nn.Sequential(
            nn.Linear(raw_dim, model_dim), nn.GELU(), nn.LayerNorm(model_dim)
        )
        self.task_embedding = nn.Embedding(num_tasks, model_dim)
        self.position_embedding = nn.Embedding(execution_horizon, model_dim)
        if self.variant != DiscriminatorVariant.POINTWISE:
            layer = nn.TransformerEncoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        else:
            self.temporal = nn.Identity()
        self.logit_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        task_ids: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if states.ndim != 3 or actions.ndim != 3:
            raise ValueError("states/actions must be batched sequences")
        if states.shape[1] != actions.shape[1] + 1:
            raise ValueError("the executed path must contain exactly H actions and H+1 states")
        horizon = actions.shape[1]
        if horizon > self.execution_horizon:
            raise ValueError("path is longer than configured execution_horizon")
        states, actions = self.normalizer(states, actions)
        current = states[:, :-1]
        following = states[:, 1:]
        change = following - current
        token = self.transition_projection(torch.cat((current, actions, change, following), dim=-1))
        position = self.position_embedding(torch.arange(horizon, device=states.device))[None]
        token = token + position + self.task_embedding(task_ids)[:, None]
        if self.variant != DiscriminatorVariant.POINTWISE:
            causal_mask = torch.triu(
                torch.ones(horizon, horizon, dtype=torch.bool, device=states.device), diagonal=1
            )
            padding_mask = None if valid_mask is None else ~valid_mask.bool()
            token = self.temporal(token, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.logit_head(token).squeeze(-1)
        if valid_mask is not None:
            logits = logits.masked_fill(~valid_mask.bool(), 0.0)
        if self.variant == DiscriminatorVariant.FINAL:
            if valid_mask is None:
                return logits[:, -1:]
            last = valid_mask.long().sum(dim=1).sub(1).clamp_min(0)
            return logits.gather(1, last[:, None])
        return logits

    @staticmethod
    def incremental_mismatch(prefix_logits: Tensor) -> Tensor:
        zero = torch.zeros_like(prefix_logits[:, :1])
        return prefix_logits - torch.cat((zero, prefix_logits[:, :-1]), dim=1)

    @staticmethod
    def final_prefix(prefix_logits: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        if valid_mask is None:
            return prefix_logits[:, -1]
        last = valid_mask.long().sum(dim=1).sub(1).clamp_min(0)
        return prefix_logits.gather(1, last[:, None]).squeeze(1)


__all__ = ["CausalPathDiscriminator", "DiscriminatorVariant", "PathNormalizer"]
