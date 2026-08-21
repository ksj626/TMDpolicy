"""PI0.5 intermediate-feature GAN discriminator."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

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


class _LayerClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, time_dim: int) -> None:
        super().__init__()
        self.feature = nn.Linear(feature_dim, hidden_dim)
        self.time = nn.Linear(time_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, features: Tensor, time_embedding: Tensor, valid: Tensor) -> Tensor:
        if features.ndim != 3 or valid.shape != features.shape[:2]:
            raise ValueError("intermediate features and valid mask must be [B,H,C] and [B,H]")
        with torch.autocast(device_type=features.device.type, enabled=False):
            hidden = self.feature(features.float()) + self.time(time_embedding.float())[:, None]
            weights = valid.to(hidden.dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
            return self.output(pooled).squeeze(-1)


class IntermediateFeatureDiscriminator(nn.Module):
    """One FP32 classifier per fake-score layer; logits are averaged."""

    variant = "pi05_intermediate_features"

    def __init__(self, feature_dims: Mapping[int, int], *, hidden_dim: int, time_dim: int = 32) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError("feature discriminator needs at least one selected layer")
        if time_dim % 2:
            raise ValueError("time_dim must be even")
        self.selected_layers = tuple(int(index) for index in feature_dims)
        if len(set(self.selected_layers)) != len(self.selected_layers):
            raise ValueError("selected feature layers must be unique")
        self.time_dim = time_dim
        self.heads = nn.ModuleDict(
            {str(index): _LayerClassifier(int(feature_dims[index]), hidden_dim, time_dim) for index in self.selected_layers}
        )

    def layer_logits(self, features: Mapping[int, Tensor], time: Tensor, valid: Tensor) -> dict[int, Tensor]:
        if tuple(features) != self.selected_layers:
            raise ValueError(
                f"feature identities {tuple(features)} do not match configured layers {self.selected_layers}"
            )
        embedding = _time_embedding(time.float(), self.time_dim)
        return {
            index: self.heads[str(index)](features[index], embedding, valid)
            for index in self.selected_layers
        }

    def forward(self, features: Mapping[int, Tensor], time: Tensor, valid: Tensor) -> Tensor:
        return torch.stack(list(self.layer_logits(features, time, valid).values()), dim=0).mean(dim=0)

    def provenance(self, *, feature_source: str) -> dict[str, Any]:
        if feature_source != "fake_score_features":
            raise ValueError("DMD2 discriminator features must come from the fake-score suffix")
        return {
            "variant": self.variant,
            "feature_source": feature_source,
            "selected_layers": list(self.selected_layers),
            "head_aggregation": "arithmetic mean of separate layer logits/losses",
        }


__all__ = ["IntermediateFeatureDiscriminator"]
