"""Causal path discriminator and train-only window normalization."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn


class WindowNormalizer(nn.Module):
    def __init__(
        self,
        state_mean: Tensor,
        state_std: Tensor,
        action_mean: Tensor,
        action_std: Tensor,
        visual_mean: Tensor,
        visual_std: Tensor,
        *,
        fitted_samples: int,
    ) -> None:
        super().__init__()
        self.register_buffer("state_mean", state_mean.float())
        self.register_buffer("state_std", state_std.float().clamp_min(1e-6))
        self.register_buffer("action_mean", action_mean.float())
        self.register_buffer("action_std", action_std.float().clamp_min(1e-6))
        self.register_buffer("visual_mean", visual_mean.float())
        self.register_buffer("visual_std", visual_std.float().clamp_min(1e-6))
        self.fitted_samples = fitted_samples

    @classmethod
    def fit(cls, train_dataset: Any, *, max_samples: int) -> "WindowNormalizer":
        """Fit only from the combined training split; validation/test are never read."""

        count = min(len(train_dataset), max_samples)
        if count < 2:
            raise ValueError("occupancy normalization needs at least two train windows")
        indices = torch.linspace(0, len(train_dataset) - 1, count).long().tolist()
        states: list[Tensor] = []
        actions: list[Tensor] = []
        visuals: list[Tensor] = []
        for index in indices:
            item = train_dataset[index]
            valid = item["valid"].bool()
            states.append(item["state"][valid].float())
            actions.append(item["action"][valid].float())
            visuals.append(item["visual"][valid].float())

        def stats(values: list[Tensor]) -> tuple[Tensor, Tensor]:
            joined = torch.cat(values, dim=0)
            return joined.mean(dim=0), joined.std(dim=0, unbiased=False)

        state_mean, state_std = stats(states)
        action_mean, action_std = stats(actions)
        visual_mean, visual_std = stats(visuals)
        return cls(
            state_mean,
            state_std,
            action_mean,
            action_std,
            visual_mean,
            visual_std,
            fitted_samples=count,
        )

    def forward(self, state: Tensor, action: Tensor, visual: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return (
            (state - self.state_mean) / self.state_std,
            (action - self.action_mean) / self.action_std,
            (visual - self.visual_mean) / self.visual_std,
        )


class OccupancyDiscriminator(nn.Module):
    """Causal real-expert=1/student=0 path classifier."""

    def __init__(
        self,
        *,
        num_tasks: int,
        window_length: int,
        model_dim: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_length = window_length
        self.input = nn.Linear(8 + 7 + 6, model_dim)
        self.task = nn.Embedding(num_tasks, model_dim)
        self.position = nn.Embedding(window_length, model_dim)
        layer = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Linear(model_dim, 1)

    def forward(
        self,
        state: Tensor,
        action: Tensor,
        visual: Tensor,
        task_index: Tensor,
        position: Tensor,
        valid: Tensor,
    ) -> Tensor:
        if state.shape[:2] != action.shape[:2] or visual.shape[:2] != action.shape[:2]:
            raise ValueError("occupancy state/action/visual sequences must align")
        length = action.shape[1]
        hidden = self.input(torch.cat((state, action, visual), dim=-1))
        hidden = hidden + self.task(task_index.long().flatten())[:, None]
        hidden = hidden + self.position(position.long().clamp(0, self.window_length - 1))
        causal = torch.triu(torch.ones(length, length, device=hidden.device, dtype=torch.bool), diagonal=1)
        hidden = self.transformer(hidden, mask=causal, src_key_padding_mask=~valid)
        mask = valid.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.output(pooled).squeeze(-1)


__all__ = ["OccupancyDiscriminator", "WindowNormalizer"]
