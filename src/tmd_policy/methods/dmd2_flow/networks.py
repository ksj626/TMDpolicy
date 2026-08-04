"""Conditioned fake-score and adversarial action networks."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _time_embedding(time: Tensor, width: int) -> Tensor:
    if width % 2:
        raise ValueError("time embedding width must be even")
    frequency = torch.exp(
        torch.linspace(0.0, -math.log(10_000.0), width // 2, device=time.device, dtype=time.dtype)
    )
    phase = time[:, None] * frequency[None]
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class ActionScoreTransformer(nn.Module):
    """Lightweight fake velocity; useful but explicitly non-paper-faithful."""

    def __init__(
        self,
        *,
        action_dim: int = 32,
        state_dim: int = 8,
        num_tasks: int = 40,
        model_dim: int = 256,
        layers: int = 4,
        heads: int = 8,
        feedforward_dim: int = 1024,
        horizon: int = 50,
    ) -> None:
        super().__init__()
        self.time_width = 32
        self.action = nn.Linear(action_dim, model_dim)
        self.state = nn.Linear(state_dim, model_dim)
        self.time = nn.Linear(self.time_width, model_dim)
        self.task = nn.Embedding(num_tasks, model_dim)
        self.position = nn.Embedding(horizon, model_dim)
        layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            feedforward_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Linear(model_dim, action_dim)

    def forward(self, x_t: Tensor, time: Tensor, state: Tensor, task_index: Tensor) -> Tensor:
        horizon = x_t.shape[1]
        hidden = self.action(x_t)
        hidden = hidden + self.state(state[..., :8])[:, None]
        hidden = hidden + self.time(_time_embedding(time, self.time_width))[:, None]
        hidden = hidden + self.task(task_index.long().flatten())[:, None]
        hidden = hidden + self.position(torch.arange(horizon, device=x_t.device))[None]
        causal = torch.triu(torch.ones(horizon, horizon, device=x_t.device, dtype=torch.bool), diagonal=1)
        return self.output(self.transformer(hidden, mask=causal))


class ActionChunkDiscriminator(nn.Module):
    """Task/observation-aligned real=1, generated=0 adversarial critic."""

    def __init__(
        self,
        *,
        action_dim: int = 7,
        state_dim: int = 8,
        num_tasks: int = 40,
        model_dim: int = 192,
        layers: int = 3,
        heads: int = 6,
        horizon: int = 50,
    ) -> None:
        super().__init__()
        self.action = nn.Linear(action_dim, model_dim)
        self.state = nn.Linear(state_dim, model_dim)
        self.task = nn.Embedding(num_tasks, model_dim)
        self.position = nn.Embedding(horizon, model_dim)
        layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            4 * model_dim,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Linear(model_dim, 1)

    def forward(self, actions: Tensor, state: Tensor, task_index: Tensor, valid: Tensor) -> Tensor:
        horizon = actions.shape[1]
        hidden = self.action(actions) + self.state(state[..., :8])[:, None]
        hidden = hidden + self.task(task_index.long().flatten())[:, None]
        hidden = hidden + self.position(torch.arange(horizon, device=actions.device))[None]
        causal = torch.triu(torch.ones(horizon, horizon, device=actions.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(hidden, mask=causal, src_key_padding_mask=~valid)
        weights = valid.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.output(pooled).squeeze(-1)


__all__ = ["ActionChunkDiscriminator", "ActionScoreTransformer"]
