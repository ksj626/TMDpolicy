"""Primary split-transformer port and explicitly adapted GRU baseline."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _time_embedding(time: Tensor, width: int) -> Tensor:
    half = width // 2
    frequencies = torch.exp(
        torch.linspace(0.0, -math.log(10_000.0), half, device=time.device, dtype=time.dtype)
    )
    phase = time[:, None] * frequencies[None]
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class _TransformerBlock(nn.Module):
    """Forward-AD-safe transformer block used by the MeanFlow JVP."""

    def __init__(
        self, model_dim: int, heads: int, feedforward_dim: int, dropout: float, *, causal: bool
    ) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = model_dim // heads
        self.attention_norm = nn.LayerNorm(model_dim)
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.attention_out = nn.Linear(model_dim, model_dim)
        self.feedforward_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, model_dim),
            nn.Dropout(dropout),
        )
        self.causal = causal

    def forward(self, value: Tensor) -> Tensor:
        batch, length, width = value.shape
        normalized = self.attention_norm(value)
        qkv = self.qkv(normalized).reshape(batch, length, 3, self.heads, self.head_dim)
        query, key, content = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        content = content.transpose(1, 2)
        logits = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        if self.causal:
            mask = torch.triu(
                torch.ones(length, length, device=value.device, dtype=torch.bool), diagonal=1
            )
            logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
        weights = F.softmax(logits, dim=-1)
        attended = (weights @ content).transpose(1, 2).reshape(batch, length, width)
        value = value + self.attention_out(attended)
        return value + self.feedforward(self.feedforward_norm(value))


class SplitTransformerMeanFlowHead(nn.Module):
    """Paper-closest SmolVLA action-space port; not an exact paper architecture.

    SmolVLA's early action-expert computation is run once by the backbone. This
    transformer receives its final action-token features and is the only module
    rerun along the inner flow. Actual early/last expert-block identities are
    recorded by `TMDStage1Program`; LeRobot 0.6.1 has no partial-expert public API,
    so this repository-owned flow head is an explicit cross-architecture port.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        context_dim: int,
        model_dim: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        horizon: int = 50,
        dropout: float = 0.0,
        attention_mode: str = "bidirectional",
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("split-transformer model_dim must be divisible by heads")
        if attention_mode not in {"bidirectional", "causal"}:
            raise ValueError("attention_mode must be bidirectional or causal")
        self.attention_mode = attention_mode
        self.input_projection = nn.Linear(action_dim + context_dim, model_dim)
        self.time_projection = nn.Linear(4, model_dim)
        self.position = nn.Embedding(horizon, model_dim)
        self.transformer = nn.ModuleList(
            [
                _TransformerBlock(
                    model_dim,
                    heads,
                    feedforward_dim,
                    dropout,
                    causal=attention_mode == "causal",
                )
                for _ in range(layers)
            ]
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.output_projection = nn.Linear(model_dim, action_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, y_s: Tensor, s: Tensor, r: Tensor, context: Tensor) -> Tensor:
        # Exact MeanFlow JVPs and their adaptive squared norms are evaluated in
        # FP32 even when the surrounding VLA training step uses BF16 autocast.
        with torch.autocast(device_type=y_s.device.type, enabled=False):
            y_s = y_s.float()
            context = context.float()
            s, r = s.float(), r.float()
            _, horizon, _ = y_s.shape
            times = torch.cat((_time_embedding(s, 2), _time_embedding(r, 2)), dim=-1)
            hidden = self.input_projection(torch.cat((y_s, context), dim=-1))
            hidden = hidden + self.time_projection(times)[:, None]
            hidden = hidden + self.position(torch.arange(horizon, device=y_s.device))[None]
            for block in self.transformer:
                hidden = block(hidden)
            return self.output_projection(self.output_norm(hidden))


class GRUMeanFlowHead(nn.Module):
    """Lightweight architectural adaptation (`tmd_gru_head`), not paper-faithful."""

    def __init__(self, *, action_dim: int, context_dim: int, hidden_dim: int, layers: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(action_dim + context_dim + 4, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, layers, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, y_s: Tensor, s: Tensor, r: Tensor, context: Tensor) -> Tensor:
        with torch.autocast(device_type=y_s.device.type, enabled=False):
            y_s = y_s.float()
            context = context.float()
            s, r = s.float(), r.float()
            times = torch.cat((_time_embedding(s, 2), _time_embedding(r, 2)), dim=-1)
            values = torch.cat(
                (y_s, context, times[:, None].expand(-1, y_s.shape[1], -1)), dim=-1
            )
            hidden, _ = self.gru(torch.nn.functional.silu(self.input_projection(values)))
            return self.output_projection(hidden)


__all__ = ["GRUMeanFlowHead", "SplitTransformerMeanFlowHead"]
