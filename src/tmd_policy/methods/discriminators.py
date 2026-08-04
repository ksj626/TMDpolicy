"""Distinct paper-feature and cached-VLA noised-action discriminators."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
        self.output = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: Tensor, time_embedding: Tensor, valid: Tensor) -> Tensor:
        if features.ndim != 3 or valid.shape != features.shape[:2]:
            raise ValueError("intermediate features and valid mask must be [B,H,C] and [B,H]")
        hidden = self.feature(features.float()) + self.time(time_embedding.float())[:, None]
        weights = valid.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.output(pooled).squeeze(-1)


class IntermediateFeatureDiscriminator(nn.Module):
    """One classifier per explicitly selected denoiser layer; logits are averaged."""

    variant = "pi05_intermediate_features"

    def __init__(self, feature_dims: Mapping[int, int], *, hidden_dim: int, time_dim: int = 32) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError("paper-feature discriminator needs at least one selected layer")
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
        logits = self.layer_logits(features, time, valid)
        return torch.stack(list(logits.values()), dim=0).mean(dim=0)

    def provenance(self, *, feature_source: str) -> dict[str, Any]:
        if feature_source not in {"fake_score_features", "teacher_features"}:
            raise ValueError("feature_source must identify fake-score or teacher features")
        return {
            "variant": self.variant,
            "feature_source": feature_source,
            "selected_layers": list(self.selected_layers),
            "head_aggregation": "arithmetic mean of separate layer logits/losses",
        }


class CachedVLAFeatureDiscriminator(nn.Module):
    """Explicit efficient adaptation ``D(a_tau,tau,c_VLA,task)``."""

    variant = "cached_vla_features"

    def __init__(
        self,
        *,
        condition_dim: int,
        action_dim: int = 32,
        num_tasks: int = 40,
        model_dim: int = 256,
        layers: int = 3,
        heads: int = 8,
        horizon: int = 50,
        time_dim: int = 32,
    ) -> None:
        super().__init__()
        self.time_dim = time_dim
        self.action = nn.Linear(action_dim, model_dim)
        self.condition = nn.Linear(condition_dim, model_dim)
        self.task = nn.Embedding(num_tasks, model_dim)
        self.time = nn.Linear(time_dim, model_dim)
        self.position = nn.Embedding(horizon, model_dim)
        layer = nn.TransformerEncoderLayer(
            model_dim, heads, 4 * model_dim, batch_first=True, norm_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.output = nn.Linear(model_dim, 1)

    def forward(
        self,
        noised_action: Tensor,
        time: Tensor,
        condition_features: Tensor,
        task_index: Tensor,
        valid: Tensor,
    ) -> Tensor:
        if noised_action.ndim != 3 or valid.shape != noised_action.shape[:2]:
            raise ValueError("cached discriminator action/valid tensors must be [B,H,D]/[B,H]")
        hidden = self.action(noised_action)
        hidden = hidden + self.condition(condition_features.detach())[:, None]
        hidden = hidden + self.task(task_index.long().flatten())[:, None]
        hidden = hidden + self.time(_time_embedding(time.float(), self.time_dim))[:, None]
        hidden = hidden + self.position(torch.arange(noised_action.shape[1], device=noised_action.device))[None]
        hidden = self.transformer(hidden, src_key_padding_mask=~valid)
        mask = valid.to(hidden.dtype).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.output(pooled).squeeze(-1)

    @staticmethod
    def provenance(cache_identity: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "variant": CachedVLAFeatureDiscriminator.variant,
            "classification": "VLA-specific cached-condition adaptation; not paper-feature reproduction",
            "condition_cache": dict(cache_identity),
        }


@dataclass(frozen=True)
class DiscriminatorBatch:
    clean: Tensor
    noised: Tensor
    noise: Tensor
    time: Tensor


__all__ = [
    "CachedVLAFeatureDiscriminator",
    "DiscriminatorBatch",
    "IntermediateFeatureDiscriminator",
]
