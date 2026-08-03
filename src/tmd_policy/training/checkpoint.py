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

CHECKPOINT_FORMAT_VERSION = 2
REQUIRED_METADATA = {
    "base_checkpoint",
    "base_revision",
    "teacher_checkpoint",
    "teacher_revision",
    "lerobot_commit",
    "dataset_revision",
    "processor_metadata",
    "training_round",
    "policy_version",
    "replay_manifest_cursor",
    "resolved_config",
    "outer_steps",
    "inner_steps",
    "inner_source_mode",
    "architecture",
}


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        saved = list(state["torch_cuda"])
        if len(saved) != torch.cuda.device_count():
            raise RuntimeError(
                "checkpoint CUDA RNG device count differs from the current process; "
                f"saved={len(saved)}, current={torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all(saved)


def _trainable_parameter_state(policy: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }


def _load_trainable_parameter_state(policy: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(policy.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise RuntimeError(f"checkpoint trainable parameters do not exist in policy: {missing}")
    current_trainable = {name for name, parameter in parameters.items() if parameter.requires_grad}
    if current_trainable != set(state):
        absent = sorted(current_trainable - set(state))
        unexpected = sorted(set(state) - current_trainable)
        raise RuntimeError(
            "trainable parameter policy differs from checkpoint; "
            f"missing={absent}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            if parameter.shape != value.shape:
                raise RuntimeError(
                    f"trainable parameter {name} shape differs: {parameter.shape} != {value.shape}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _state_or_none(value: Any | None) -> Any | None:
    return None if value is None else value.state_dict()


def save_training_checkpoint(
    path: str | Path,
    *,
    policy: nn.Module,
    discriminator: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically save every mutable state needed for deterministic resume."""

    missing = sorted(REQUIRED_METADATA - set(metadata))
    if missing:
        raise ValueError(f"checkpoint metadata is incomplete: {missing}")
    generator = getattr(policy, "generator", None)
    transition_head = getattr(generator, "transition_head", None)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": dict(metadata),
        "transition_head": None if transition_head is None else transition_head.state_dict(),
        "policy_trainable_parameters": _trainable_parameter_state(policy),
        "discriminator": None if discriminator is None else discriminator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": _state_or_none(scheduler),
        "amp_scaler": _state_or_none(scaler),
        "rng": capture_rng_state(),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}-", suffix=".tmp", dir=target.parent
    )
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


def load_training_checkpoint(
    path: str | Path,
    *,
    policy: nn.Module,
    discriminator: nn.Module | None,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    scaler: Any | None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Restore mutable state and return immutable run metadata."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError(
            f"unsupported checkpoint format {payload.get('format_version')!r}; "
            f"expected {CHECKPOINT_FORMAT_VERSION}"
        )
    metadata = payload.get("metadata", {})
    missing = sorted(REQUIRED_METADATA - set(metadata))
    if missing:
        raise RuntimeError(f"checkpoint metadata is incomplete: {missing}")
    generator = getattr(policy, "generator", None)
    transition_head = getattr(generator, "transition_head", None)
    if payload["transition_head"] is not None:
        if transition_head is None:
            raise RuntimeError("checkpoint contains a transition head but policy does not")
        transition_head.load_state_dict(payload["transition_head"], strict=True)
    _load_trainable_parameter_state(policy, payload["policy_trainable_parameters"])
    if payload["discriminator"] is not None:
        if discriminator is None:
            raise RuntimeError("checkpoint contains a discriminator but none was supplied")
        discriminator.load_state_dict(payload["discriminator"], strict=True)
    elif discriminator is not None:
        raise RuntimeError("a discriminator was supplied but checkpoint has none")
    optimizer.load_state_dict(payload["optimizer"])
    if payload["scheduler"] is not None:
        if scheduler is None:
            raise RuntimeError("checkpoint contains a scheduler but none was supplied")
        scheduler.load_state_dict(payload["scheduler"])
    elif scheduler is not None:
        raise RuntimeError("a scheduler was supplied but checkpoint has none")
    if payload["amp_scaler"] is not None:
        if scaler is None:
            raise RuntimeError("checkpoint contains an AMP scaler but none was supplied")
        scaler.load_state_dict(payload["amp_scaler"])
    elif scaler is not None:
        raise RuntimeError("an AMP scaler was supplied but checkpoint has none")
    if restore_rng:
        restore_rng_state(payload["rng"])
    return dict(metadata)


def load_policy_for_inference(path: str | Path, policy: nn.Module) -> dict[str, Any]:
    """Load only audited policy weights and immutable metadata for evaluation."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise RuntimeError("unsupported checkpoint format")
    metadata = payload.get("metadata", {})
    missing = sorted(REQUIRED_METADATA - set(metadata))
    if missing:
        raise RuntimeError(f"checkpoint metadata is incomplete: {missing}")
    generator = getattr(policy, "generator", None)
    transition_head = getattr(generator, "transition_head", None)
    if payload["transition_head"] is not None:
        if transition_head is None:
            raise RuntimeError("checkpoint contains a transition head but policy does not")
        transition_head.load_state_dict(payload["transition_head"], strict=True)
    _load_trainable_parameter_state(policy, payload["policy_trainable_parameters"])
    return dict(metadata)


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "REQUIRED_METADATA",
    "capture_rng_state",
    "load_policy_for_inference",
    "load_training_checkpoint",
    "restore_rng_state",
    "save_training_checkpoint",
]
