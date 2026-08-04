from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"])
    if torch.cuda.is_available() and value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])


def save_method_checkpoint(
    path: str | Path,
    *,
    method_name: str,
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer],
    schedulers: Mapping[str, Any],
    scaler: Any | None,
    counters: Mapping[str, int],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    trainable_names: Mapping[str, list[str]],
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 3,
        "method_name": method_name,
        "models": {name: model.state_dict() for name, model in models.items()},
        "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
        "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
        "scaler": scaler.state_dict() if scaler is not None else None,
        "counters": dict(counters),
        "config": dict(config),
        "provenance": dict(provenance),
        "trainable_names": {name: sorted(names) for name, names in trainable_names.items()},
        "rng": _rng_state(),
    }
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def load_method_checkpoint(
    path: str | Path,
    *,
    expected_method: str,
    models: Mapping[str, nn.Module],
    optimizers: Mapping[str, torch.optim.Optimizer] | None = None,
    schedulers: Mapping[str, Any] | None = None,
    scaler: Any | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    value = torch.load(Path(path), map_location="cpu", weights_only=False)
    if value.get("format_version") != 3 or value.get("method_name") != expected_method:
        raise RuntimeError(
            f"checkpoint is not format-v3 state for {expected_method}: "
            f"{value.get('format_version')}/{value.get('method_name')}"
        )
    expected_models = set(models)
    if set(value["models"]) != expected_models:
        raise RuntimeError(f"checkpoint model components mismatch: {set(value['models'])} != {expected_models}")
    for name, model in models.items():
        model.load_state_dict(value["models"][name], strict=True)
    if optimizers is not None:
        if set(value["optimizers"]) != set(optimizers):
            raise RuntimeError("checkpoint optimizer components mismatch")
        for name, optimizer in optimizers.items():
            optimizer.load_state_dict(value["optimizers"][name])
    if schedulers is not None:
        if set(value["schedulers"]) != set(schedulers):
            raise RuntimeError("checkpoint scheduler components mismatch")
        for name, scheduler in schedulers.items():
            scheduler.load_state_dict(value["schedulers"][name])
    if scaler is not None and value["scaler"] is not None:
        scaler.load_state_dict(value["scaler"])
    if restore_rng:
        _restore_rng(value["rng"])
    return {
        key: value[key]
        for key in ("counters", "config", "provenance", "trainable_names")
    }
