"""Deterministic multi-phase trainer with exact boundary checkpoint resume."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Sampler

from tmd_policy.backends.lerobot.compatibility import verify_installed_lerobot
from tmd_policy.config import project_path, save_resolved_config


class DeterministicBatchSampler(Sampler[list[int]]):
    """Permutation is a pure function of seed/epoch; resume skips consumed batches."""

    def __init__(
        self,
        size: int,
        batch_size: int,
        *,
        seed: int,
        epoch: int = 0,
        start_batch: int = 0,
        drop_last: bool = True,
    ) -> None:
        if size < 1 or batch_size < 1:
            raise ValueError("dataset and batch size must be positive")
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = epoch
        self.start_batch = start_batch
        self.drop_last = drop_last

    @property
    def batches_per_epoch(self) -> int:
        return self.size // self.batch_size if self.drop_last else math.ceil(self.size / self.batch_size)

    def __len__(self) -> int:
        return max(0, self.batches_per_epoch - self.start_batch)

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch * 1_000_003)
        order = torch.randperm(self.size, generator=generator).tolist()
        batches = [order[index : index + self.batch_size] for index in range(0, self.size, self.batch_size)]
        if self.drop_last and batches and len(batches[-1]) != self.batch_size:
            batches.pop()
        yield from batches[self.start_batch :]

    def state_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "epoch": self.epoch,
            "start_batch": self.start_batch,
            "drop_last": self.drop_last,
        }


class TrainingProgram(nn.Module, ABC):
    """A concrete model graph and its ordered optimizer phases."""

    @abstractmethod
    def phase_schedule(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def loss(self, batch: dict[str, Any], phase: str) -> tuple[Tensor, dict[str, float]]:
        raise NotImplementedError

    @abstractmethod
    def make_optimizers(self, training: dict[str, Any]) -> dict[str, Optimizer]:
        raise NotImplementedError

    def validation_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        phase = self.phase_schedule()[-1]
        return self.loss(batch, phase)

    def extra_provenance(self) -> dict[str, Any]:
        return {}

    def trainable_parameter_names(self) -> list[str]:
        return [name for name, parameter in self.named_parameters() if parameter.requires_grad]

    def validate_phase_gradients(self, phase: str) -> None:
        """Fail fast on method-specific gradient-path invariants."""


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
        return {"repository": str(root), "commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"repository": str(root), "commit": None, "dirty": None}


def _scheduler(optimizer: Optimizer, training: dict[str, Any]) -> LRScheduler:
    warmup = int(training.get("warmup_steps", 0))
    total = int(training["max_steps"])

    def factor(step: int) -> float:
        if warmup and step < warmup:
            return max(1e-8, (step + 1) / warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return max(float(training.get("minimum_lr_scale", 0.05)), 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _autocast(device: torch.device, precision: str):
    enabled = precision != "no" and device.type == "cuda"
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _checkpoint_payload(
    program: TrainingProgram,
    optimizers: dict[str, Optimizer],
    schedulers: dict[str, LRScheduler],
    scaler: torch.amp.GradScaler,
    *,
    config: dict[str, Any],
    provenance: dict[str, Any],
    global_step: int,
    epoch: int,
    next_batch: int,
    sampler: Sampler[list[int]],
) -> dict[str, Any]:
    sampler_state = dict(sampler.state_dict()) if hasattr(sampler, "state_dict") else {}
    sampler_state.update(
        {
            "type": f"{type(sampler).__module__}.{type(sampler).__qualname__}",
            "epoch": epoch,
            "start_batch": next_batch,
            "resume_contract": "pure function of sampler type/config, seed, epoch, and batch cursor",
        }
    )
    return {
        "format": "tmdpolicy.training/v1",
        "program": program.state_dict(),
        "optimizers": {name: value.state_dict() for name, value in optimizers.items()},
        "schedulers": {name: value.state_dict() for name, value in schedulers.items()},
        "scaler": scaler.state_dict(),
        "counters": {"global_step": global_step, "epoch": epoch, "next_batch": next_batch},
        "sampler": sampler_state,
        "rng": _rng_state(),
        "config": config,
        "provenance": provenance,
        "trainable_parameter_names": program.trainable_parameter_names(),
    }


def _atomic_save(payload: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def _load_checkpoint(
    path: Path,
    program: TrainingProgram,
    optimizers: dict[str, Optimizer],
    schedulers: dict[str, LRScheduler],
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    sampler: Sampler[list[int]],
) -> tuple[int, int, int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "tmdpolicy.training/v1":
        raise ValueError(f"unsupported checkpoint format in {path}")
    previous = dict(payload["config"])
    current = dict(config)
    previous.pop("_config_path", None)
    current.pop("_config_path", None)
    if previous != current:
        raise ValueError("resume configuration differs from the checkpoint's fully resolved configuration")
    program.load_state_dict(payload["program"], strict=True)
    if set(payload["optimizers"]) != set(optimizers):
        raise ValueError("optimizer phase set changed across resume")
    for name, optimizer in optimizers.items():
        optimizer.load_state_dict(payload["optimizers"][name])
        schedulers[name].load_state_dict(payload["schedulers"][name])
    scaler.load_state_dict(payload["scaler"])
    if payload["trainable_parameter_names"] != program.trainable_parameter_names():
        raise ValueError("trainable parameter identities changed across resume")
    stored_sampler = dict(payload.get("sampler", {}))
    current_sampler = dict(sampler.state_dict()) if hasattr(sampler, "state_dict") else {}
    current_sampler["type"] = f"{type(sampler).__module__}.{type(sampler).__qualname__}"
    for transient in ("epoch", "start_batch", "resume_contract"):
        stored_sampler.pop(transient, None)
        current_sampler.pop(transient, None)
    if stored_sampler != current_sampler:
        raise ValueError(
            f"sampler identity/configuration changed across resume: {stored_sampler} != {current_sampler}"
        )
    _restore_rng(payload["rng"])
    counters = payload["counters"]
    return int(counters["global_step"]), int(counters["epoch"]), int(counters["next_batch"])


@torch.no_grad()
def _validate(
    program: TrainingProgram,
    dataset: Dataset,
    *,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    precision: str,
    max_batches: int,
) -> dict[str, float]:
    program.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    losses: list[float] = []
    metrics: dict[str, list[float]] = {}
    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        with _autocast(device, precision):
            loss, values = program.validation_loss(batch)
        losses.append(float(loss.detach()))
        for name, value in values.items():
            metrics.setdefault(name, []).append(float(value))
    program.train()
    return {
        "validation/loss": float(np.mean(losses)),
        **{f"validation/{name}": float(np.mean(values)) for name, values in metrics.items()},
    }


def run_training(
    program: TrainingProgram,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    *,
    config: dict[str, Any],
    output_dir: str | Path,
    resume: str | Path | None = None,
    train_batch_sampler: Sampler[list[int]] | None = None,
) -> dict[str, Any]:
    """Run real DataLoader/model updates; no dry-run branch exists."""

    training = config["training"]
    device = torch.device(training["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training was requested but torch.cuda.is_available() is false")
    output = Path(output_dir)
    if resume is None:
        output.mkdir(parents=True, exist_ok=False)
    else:
        output.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output / "resolved_config.yaml")
    seed_everything(int(training["seed"]))
    program.to(device)
    names = program.trainable_parameter_names()
    if not names:
        raise RuntimeError("training program contains no trainable parameters")
    (output / "trainable_parameters.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")

    optimizers = program.make_optimizers(training)
    if set(optimizers) != set(program.phase_schedule()):
        raise RuntimeError("phase_schedule and optimizer names must contain the same unique phases")
    schedulers = {name: _scheduler(value, training) for name, value in optimizers.items()}
    scaler = torch.amp.GradScaler("cuda", enabled=training["mixed_precision"] == "fp16" and device.type == "cuda")
    compatibility = verify_installed_lerobot(
        expected_version=config["backend"]["lerobot_version"],
        expected_source_hashes=config["backend"].get("expected_source_hashes"),
    )
    provenance = {
        "created_unix_s": time.time(),
        "git": _git_provenance(),
        "lerobot": compatibility,
        "models": config["models"],
        "dataset": config["dataset"],
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_devices": [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        "manifest": {
            "path": str(project_path(config["dataset"]["manifest"]).resolve()),
            "sha256": _sha256(project_path(config["dataset"]["manifest"])),
        },
        **program.extra_provenance(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    accumulation = int(training["gradient_accumulation"])
    sampler = train_batch_sampler or DeterministicBatchSampler(
        len(train_dataset), int(training["batch_size"]), seed=int(training["seed"]), drop_last=True
    )
    if not hasattr(sampler, "epoch") or not hasattr(sampler, "start_batch"):
        raise TypeError("training batch samplers must expose epoch and start_batch for exact resume")

    global_step = epoch = next_batch = 0
    if resume is not None:
        global_step, epoch, next_batch = _load_checkpoint(
            Path(resume), program, optimizers, schedulers, scaler, config, sampler
        )
    metrics_path = output / "metrics.jsonl"
    clip = float(training["gradient_clip_norm"])

    while global_step < int(training["max_steps"]):
        sampler.epoch = epoch
        sampler.start_batch = next_batch
        loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=int(training.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
            persistent_workers=bool(training.get("num_workers", 0)),
        )
        iterator = iter(loader)
        while global_step < int(training["max_steps"]):
            microbatches = []
            for _ in range(accumulation):
                try:
                    microbatches.append(next(iterator))
                except StopIteration:
                    break
            if len(microbatches) < accumulation:
                epoch += 1
                next_batch = 0
                break
            step_metrics: dict[str, float] = {}
            for phase in program.phase_schedule():
                optimizer = optimizers[phase]
                optimizer.zero_grad(set_to_none=True)
                phase_losses = []
                phase_values: dict[str, list[float]] = {}
                for batch in microbatches:
                    with _autocast(device, training["mixed_precision"]):
                        loss, values = program.loss(batch, phase)
                        scaled_loss = loss / accumulation
                    scaler.scale(scaled_loss).backward()
                    phase_losses.append(float(loss.detach()))
                    for name, value in values.items():
                        phase_values.setdefault(name, []).append(float(value))
                scaler.unscale_(optimizer)
                program.validate_phase_gradients(phase)
                parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
                scaler.step(optimizer)
                scaler.update()
                schedulers[phase].step()
                step_metrics[f"train/{phase}/loss"] = float(np.mean(phase_losses))
                step_metrics[f"train/{phase}/gradient_norm"] = float(gradient_norm)
                step_metrics[f"train/{phase}/learning_rate"] = float(optimizer.param_groups[0]["lr"])
                for name, values in phase_values.items():
                    step_metrics[f"train/{phase}/{name}"] = float(np.mean(values))
            global_step += 1
            next_batch += accumulation
            row: dict[str, Any] = {"global_step": global_step, "epoch": epoch, **step_metrics}
            if global_step % int(training["validation_interval"]) == 0:
                row.update(
                    _validate(
                        program,
                        validation_dataset,
                        batch_size=int(training["batch_size"]),
                        num_workers=int(training.get("num_workers", 0)),
                        device=device,
                        precision=training["mixed_precision"],
                        max_batches=int(training.get("validation_batches", 8)),
                    )
                )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if global_step % int(training["checkpoint_interval"]) == 0:
                payload = _checkpoint_payload(
                    program,
                    optimizers,
                    schedulers,
                    scaler,
                    config=config,
                    provenance=provenance,
                    global_step=global_step,
                    epoch=epoch,
                    next_batch=next_batch,
                    sampler=sampler,
                )
                checkpoint = output / "checkpoints" / f"step-{global_step:08d}.pt"
                _atomic_save(payload, checkpoint)
                _atomic_save(payload, output / "checkpoints" / "latest.pt")

    final_payload = _checkpoint_payload(
        program,
        optimizers,
        schedulers,
        scaler,
        config=config,
        provenance=provenance,
        global_step=global_step,
        epoch=epoch,
        next_batch=next_batch,
        sampler=sampler,
    )
    final_checkpoint = output / "checkpoints" / "final.pt"
    _atomic_save(final_payload, final_checkpoint)
    report = {
        "method": config["method"],
        "global_step": global_step,
        "epoch": epoch,
        "checkpoint": str(final_checkpoint.resolve()),
        "trainable_parameters": len(names),
        "metrics": str(metrics_path.resolve()),
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = ["DeterministicBatchSampler", "TrainingProgram", "run_training", "seed_everything"]
