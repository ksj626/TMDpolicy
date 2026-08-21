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
from collections import Counter
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm

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

    def backward_loss_scale(self, phase: str) -> float:
        """Return a numerically safe scale that is removed before the optimizer step."""

        return 1.0

    def inference_state_dict(self) -> dict[str, Tensor] | None:
        """Return a minimal evaluable state, or ``None`` when unsupported."""

        return None

    def set_step_context(self, **values: Any) -> None:
        """Receive non-persistent trainer counters for diagnostics."""

    def prepare_phase_batch(self, batch: dict[str, Any], phase: str) -> dict[str, Any]:
        """Optionally replace a phase's conditioning batch."""

        return batch

    def gradient_diagnostics(self, phase: str) -> dict[str, float]:
        return {}

    def optimizer_step_diagnostics(self, phase: str, optimizer: Optimizer) -> dict[str, float]:
        return {}


def validate_optimizer_parameter_ownership(optimizers: dict[str, Optimizer]) -> None:
    """Reject parameters registered with more than one independently stepped optimizer."""

    owners: dict[int, tuple[str, int, int]] = {}
    for optimizer_name, optimizer in optimizers.items():
        for group_index, group in enumerate(optimizer.param_groups):
            for parameter_index, parameter in enumerate(group["params"]):
                identity = id(parameter)
                if identity in owners:
                    previous_name, previous_group, previous_index = owners[identity]
                    raise RuntimeError(
                        "optimizer parameter overlap: "
                        f"{previous_name}[group={previous_group},parameter={previous_index}] and "
                        f"{optimizer_name}[group={group_index},parameter={parameter_index}] own "
                        "the same parameter"
                    )
                owners[identity] = (optimizer_name, group_index, parameter_index)


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


def _scheduler(
    optimizer: Optimizer,
    training: dict[str, Any],
    *,
    updates_per_global_step: int,
) -> LRScheduler:
    if updates_per_global_step < 1:
        raise ValueError("scheduler update multiplicity must be positive")
    # max_steps and warmup_steps are generator/global-step units. TTUR phases
    # may update several times per global step, so each optimizer gets a
    # schedule expressed in its own actual update count.
    warmup = int(training.get("warmup_steps", 0)) * updates_per_global_step
    total = int(training["max_steps"]) * updates_per_global_step

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
    optimizer_updates: dict[str, int],
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
        "counters": {
            "global_step": global_step,
            "epoch": epoch,
            "next_batch": next_batch,
            "optimizer_updates": dict(optimizer_updates),
        },
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


def _inference_checkpoint_payload(
    program: TrainingProgram,
    *,
    config: dict[str, Any],
    provenance: dict[str, Any],
    global_step: int,
    epoch: int,
) -> dict[str, Any] | None:
    state = program.inference_state_dict()
    if state is None:
        return None
    return {
        "format": "tmdpolicy.inference/v1",
        "program": state,
        "counters": {"global_step": global_step, "epoch": epoch},
        "config": config,
        "provenance": provenance,
        "trainable_parameter_names": sorted(state),
        "base_model_contract": "immutable hub revision plus saved trainable delta",
    }


def _load_metric_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or "global_step" not in value:
                raise ValueError(f"invalid metrics row at {path}:{line_number}")
            rows.append(value)
    return rows


def _loss_series(rows: list[dict[str, Any]]) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        step = int(row["global_step"])
        for name, value in row.items():
            if (
                name.endswith("/loss")
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
            ):
                series.setdefault(name, []).append((step, float(value)))
    return series


def _append_loss_series(
    series: dict[str, list[tuple[int, float]]], row: dict[str, Any]
) -> None:
    step = int(row["global_step"])
    for name, value in row.items():
        if (
            name.endswith("/loss")
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
        ):
            series.setdefault(name, []).append((step, float(value)))


def _write_training_progress(
    series: dict[str, list[tuple[int, float]]], target: Path, *, method: str
) -> None:
    """Atomically refresh a standalone plot of all persisted loss series."""

    if not series:
        return
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for name, all_points in sorted(series.items()):
        # A PNG cannot display every point from a 100k-step run. Bound redraw
        # cost while retaining the first, last, and regularly sampled history.
        stride = max(1, math.ceil(len(all_points) / 2_000))
        points = all_points[::stride]
        if points and points[-1] != all_points[-1]:
            points.append(all_points[-1])
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                label=name,
                linewidth=1.5,
            )
    axis.set_title(f"{method} training progress")
    axis.set_xlabel("global step")
    axis.set_ylabel("loss")
    axis.set_yscale("symlog", linthresh=1.0e-4)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.partial{target.suffix}")
    try:
        figure.savefig(temporary, format="png", dpi=150)
        os.replace(temporary, target)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def _load_checkpoint(
    path: Path,
    program: TrainingProgram,
    optimizers: dict[str, Optimizer],
    schedulers: dict[str, LRScheduler],
    scaler: torch.amp.GradScaler,
    config: dict[str, Any],
    sampler: Sampler[list[int]],
) -> tuple[int, int, int, dict[str, int]]:
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
    global_step = int(counters["global_step"])
    stored_updates = counters.get("optimizer_updates")
    if stored_updates is None:
        multiplicity = Counter(program.phase_schedule())
        stored_updates = {name: global_step * multiplicity[name] for name in optimizers}
    optimizer_updates = {name: int(stored_updates[name]) for name in optimizers}
    return global_step, int(counters["epoch"]), int(counters["next_batch"]), optimizer_updates


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
    validation_batches = min(max_batches, len(loader))
    batches = tqdm(
        islice(loader, validation_batches),
        total=validation_batches,
        desc="validation",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    for batch in batches:
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
    initial_rollout_replay: str | Path | None = None,
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
    phase_schedule = program.phase_schedule()
    if set(optimizers) != set(phase_schedule):
        raise RuntimeError("phase_schedule and optimizer names must contain the same unique phases")
    validate_optimizer_parameter_ownership(optimizers)
    phase_counts = Counter(phase_schedule)
    schedulers = {
        name: _scheduler(
            value,
            training,
            updates_per_global_step=phase_counts[name],
        )
        for name, value in optimizers.items()
    }
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
    optimizer_updates = {name: 0 for name in optimizers}
    if resume is not None:
        global_step, epoch, next_batch, optimizer_updates = _load_checkpoint(
            Path(resume), program, optimizers, schedulers, scaler, config, sampler
        )

    rollout_manager = None
    replay_settings = config.get("dmd2", {}).get("student_rollout_replay", {})
    if bool(replay_settings.get("enabled", False)):
        from tmd_policy.training.rollout_replay import AsyncStudentRolloutManager

        rollout_manager = AsyncStudentRolloutManager(
            config=config,
            output=output,
            initial_round=initial_rollout_replay,
        )
        program.set_student_replay(rollout_manager.replay)

        def rollout_snapshot() -> Path:
            payload = _inference_checkpoint_payload(
                program,
                config=config,
                provenance=provenance,
                global_step=global_step,
                epoch=epoch,
            )
            if payload is None:
                raise RuntimeError("student replay requires inference-delta checkpoints")
            target = (
                output
                / "student_rollout_replay"
                / "snapshots"
                / f"generator-{optimizer_updates.get('generator', 0):08d}.pt"
            )
            _atomic_save(payload, target)
            return target

        if rollout_manager.replay.size == 0:
            if not bool(replay_settings.get("initial_blocking", True)):
                raise RuntimeError("DMD2 fidelity requires a blocking initial balanced student rollout")
            rollout_manager.launch(
                rollout_snapshot(),
                generator_updates=optimizer_updates.get("generator", 0),
                blocking=True,
            )
        rollout_manager.write_status()
    metrics_path = output / "metrics.jsonl"
    progress_plot_path = output / "training_progress.png"
    loss_series = _loss_series(_load_metric_history(metrics_path))
    clip = float(training["gradient_clip_norm"])
    maximum_steps = int(training["max_steps"])
    progress = tqdm(
        total=maximum_steps,
        initial=global_step,
        desc=f"train {config['method']}",
        unit="step",
        dynamic_ncols=True,
    )

    while global_step < maximum_steps:
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
        while global_step < maximum_steps:
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
            step_metric_values: dict[str, list[float]] = {}
            phase_occurrences: Counter[str] = Counter()
            for phase_index, phase in enumerate(phase_schedule, start=1):
                phase_occurrences[phase] += 1
                diagnostics_interval = int(training.get("diagnostics_interval", 1))
                diagnostics_enabled = (
                    (global_step + 1) % diagnostics_interval == 0
                    and phase_occurrences[phase] == 1
                )
                program.set_step_context(
                    global_step=global_step,
                    phase=phase,
                    phase_occurrence=phase_occurrences[phase],
                    optimizer_update_count=optimizer_updates[phase],
                    diagnostics_enabled=diagnostics_enabled,
                )
                progress.set_postfix_str(
                    f"phase={phase} {phase_index}/{len(phase_schedule)} microbatch=0/{accumulation}",
                    refresh=True,
                )
                optimizer = optimizers[phase]
                optimizer.zero_grad(set_to_none=True)
                backward_scale = float(program.backward_loss_scale(phase))
                if not math.isfinite(backward_scale) or backward_scale <= 0.0:
                    raise RuntimeError(
                        f"{phase} backward loss scale must be positive and finite, got {backward_scale}"
                    )
                phase_losses = []
                phase_values: dict[str, list[float]] = {}
                for microbatch_index, batch in enumerate(microbatches, start=1):
                    phase_batch = program.prepare_phase_batch(batch, phase)
                    with _autocast(device, training["mixed_precision"]):
                        loss, values = program.loss(phase_batch, phase)
                        if loss.numel() != 1 or not torch.isfinite(loss.detach()).all():
                            raise RuntimeError(f"{phase} loss contains NaN/Inf before backward")
                        scaled_loss = loss * backward_scale / accumulation
                    scaler.scale(scaled_loss).backward()
                    phase_losses.append(float(loss.detach()))
                    for name, value in values.items():
                        phase_values.setdefault(name, []).append(float(value))
                    progress.set_postfix_str(
                        f"phase={phase} {phase_index}/{len(phase_schedule)} "
                        f"microbatch={microbatch_index}/{accumulation}",
                        refresh=True,
                    )
                scaler.unscale_(optimizer)
                if backward_scale != 1.0:
                    for group in optimizer.param_groups:
                        for parameter in group["params"]:
                            if parameter.grad is not None:
                                parameter.grad.div_(backward_scale)
                program.validate_phase_gradients(phase)
                parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all()
                    for parameter in parameters
                ):
                    raise RuntimeError(f"{phase} optimizer gradient contains NaN/Inf before step")
                phase_diagnostics = program.gradient_diagnostics(phase)
                gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer_updates[phase] += 1
                schedulers[phase].step()
                phase_diagnostics.update(program.optimizer_step_diagnostics(phase, optimizer))
                phase_diagnostics["gradient/nonfinite_fraction"] = 0.0
                phase_diagnostics["optimizer_update_count"] = float(optimizer_updates[phase])
                phase_diagnostics["scheduler_step"] = float(schedulers[phase].last_epoch)
                step_metric_values.setdefault(f"train/{phase}/loss", []).append(
                    float(np.mean(phase_losses))
                )
                step_metric_values.setdefault(f"train/{phase}/gradient_norm", []).append(
                    float(gradient_norm)
                )
                step_metric_values.setdefault(f"train/{phase}/learning_rate", []).append(
                    float(optimizer.param_groups[0]["lr"])
                )
                for name, values in phase_values.items():
                    step_metric_values.setdefault(f"train/{phase}/{name}", []).append(
                        float(np.mean(values))
                    )
                for name, value in phase_diagnostics.items():
                    step_metric_values.setdefault(f"train/{phase}/{name}", []).append(float(value))
            step_metrics = {
                name: float(np.mean(values)) for name, values in step_metric_values.items()
            }
            global_step += 1
            next_batch += accumulation
            row: dict[str, Any] = {
                "global_step": global_step,
                "epoch": epoch,
                **{
                    f"{name}_update_count": optimizer_updates[name]
                    for name in sorted(optimizer_updates)
                },
                **{
                    f"optimizer/{name}/learning_rate": float(optimizers[name].param_groups[0]["lr"])
                    for name in sorted(optimizers)
                },
                **{
                    f"optimizer/{name}/scheduler_step": int(schedulers[name].last_epoch)
                    for name in sorted(schedulers)
                },
                **step_metrics,
            }
            if rollout_manager is not None:
                rollout_manager.poll()
                generator_updates = optimizer_updates.get("generator", 0)
                if rollout_manager.should_refresh(generator_updates):
                    rollout_manager.launch(
                        rollout_snapshot(),
                        generator_updates=generator_updates,
                        blocking=False,
                    )
                rollout_manager.write_status()
                row.update(rollout_manager.metrics())
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
            _append_loss_series(loss_series, row)
            _write_training_progress(
                loss_series,
                progress_plot_path,
                method=str(config["method"]),
            )
            progress.update(1)
            progress.set_postfix(
                {
                    key.removeprefix("train/").removesuffix("/loss"): f"{value:.4g}"
                    for key, value in step_metrics.items()
                    if key.endswith("/loss")
                },
                refresh=True,
            )
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
                    optimizer_updates=optimizer_updates,
                )
                checkpoint = output / "checkpoints" / f"step-{global_step:08d}.pt"
                _atomic_save(payload, checkpoint)
                _atomic_save(payload, output / "checkpoints" / "latest.pt")
            inference_interval = int(training.get("inference_checkpoint_interval", 0))
            if inference_interval and global_step % inference_interval == 0:
                inference_payload = _inference_checkpoint_payload(
                    program,
                    config=config,
                    provenance=provenance,
                    global_step=global_step,
                    epoch=epoch,
                )
                if inference_payload is not None:
                    inference_checkpoint = (
                        output
                        / "inference_checkpoints"
                        / f"step-{global_step:08d}.pt"
                    )
                    _atomic_save(inference_payload, inference_checkpoint)
                    _atomic_save(
                        inference_payload,
                        output / "inference_checkpoints" / "latest.pt",
                    )

    progress.close()
    if rollout_manager is not None:
        rollout_manager.poll()
        rollout_manager.write_status()

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
        optimizer_updates=optimizer_updates,
    )
    final_checkpoint = output / "checkpoints" / "final.pt"
    _atomic_save(final_payload, final_checkpoint)
    final_inference_payload = _inference_checkpoint_payload(
        program,
        config=config,
        provenance=provenance,
        global_step=global_step,
        epoch=epoch,
    )
    final_inference_checkpoint = None
    if final_inference_payload is not None:
        final_inference_checkpoint = output / "inference_checkpoints" / "final.pt"
        _atomic_save(final_inference_payload, final_inference_checkpoint)
    report = {
        "method": config["method"],
        "global_step": global_step,
        "epoch": epoch,
        "optimizer_updates": optimizer_updates,
        "checkpoint": str(final_checkpoint.resolve()),
        "trainable_parameters": len(names),
        "metrics": str(metrics_path.resolve()),
        "progress_plot": str(progress_plot_path.resolve()),
        "inference_checkpoint": (
            str(final_inference_checkpoint.resolve())
            if final_inference_checkpoint is not None
            else None
        ),
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = [
    "DeterministicBatchSampler",
    "TrainingProgram",
    "run_training",
    "seed_everything",
    "validate_optimizer_parameter_ownership",
]
