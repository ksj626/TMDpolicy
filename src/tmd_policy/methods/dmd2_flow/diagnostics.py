"""Stateless DMD2 distribution and timestep diagnostics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def masked_mean(values: Tensor, valid_coordinates: Tensor) -> Tensor:
    mask = valid_coordinates.to(values.dtype)
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1)
    return (values * mask).flatten(1).sum(dim=1) / denominator


def cosine(left: Tensor, right: Tensor) -> Tensor:
    return F.cosine_similarity(left.detach().float().flatten(1), right.detach().float().flatten(1))


def time_binned_metrics(prefix: str, time: Tensor, values: Tensor, *, bins: int = 5) -> dict[str, float]:
    if time.ndim != 1 or values.shape != time.shape:
        raise ValueError("time-binned diagnostics require [B] times and values")
    indices = torch.clamp((time.detach().float() * bins).long(), max=bins - 1)
    result: dict[str, float] = {}
    for index in range(bins):
        selected = indices == index
        result[f"{prefix}/bin_{index}_fraction"] = float(selected.float().mean())
        if torch.any(selected):
            result[f"{prefix}/bin_{index}_value"] = float(values.detach().float()[selected].mean())
    return result


def distribution_metrics(prefix: str, values: Tensor) -> dict[str, float]:
    flat = values.detach().float().flatten()
    quantiles = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9], device=flat.device))
    return {
        f"{prefix}/mean": float(flat.mean()),
        f"{prefix}/std": float(flat.std(unbiased=False)),
        f"{prefix}/p10": float(quantiles[0]),
        f"{prefix}/p50": float(quantiles[1]),
        f"{prefix}/p90": float(quantiles[2]),
    }


def binary_auc(real_logits: Tensor, fake_logits: Tensor) -> float:
    real, fake = real_logits.detach().float().flatten(), fake_logits.detach().float().flatten()
    if real.numel() == 0 or fake.numel() == 0:
        return float("nan")
    comparisons = real[:, None] - fake[None, :]
    return float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean())


__all__ = ["binary_auc", "cosine", "distribution_metrics", "masked_mean", "time_binned_metrics"]
