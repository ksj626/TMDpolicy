from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tmd_policy.config import ExperimentConfig, save_resolved_config
from tmd_policy.models.smolvla_tmd import load_smolvla_tmd
from tmd_policy.training.checkpoint import save_training_checkpoint


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(config: ExperimentConfig, parameters: Any) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        parameters,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )


def _read_record(manifest: Path, index: int = 0) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if not records:
        raise ValueError(f"expert manifest contains no records: {manifest}")
    record = records[index]
    payload_path = manifest.parent / record["payload"]
    with np.load(payload_path, allow_pickle=False) as payload:
        arrays = {key: payload[key] for key in payload.files}
    return record, arrays


def _expert_batch(
    record: dict[str, Any],
    arrays: dict[str, np.ndarray],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "observation.state": torch.from_numpy(arrays["path_states"][0]).to(device)[None],
        "action": torch.from_numpy(arrays["plan_actions"]).to(device)[None],
        "action_is_pad": torch.from_numpy(~arrays["plan_valid"].astype(bool)).to(device)[None],
        "task": [record["instruction"]] * batch_size,
    }
    for key, value in arrays.items():
        if key.startswith("image::"):
            batch[key.removeprefix("image::")] = torch.from_numpy(value).to(device)[None]
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.expand(batch_size, *value.shape[1:]).clone()
    return batch


def train_expert_chunk(
    config: ExperimentConfig,
    *,
    expert_manifest: str | Path,
    output_dir: str | Path,
    record_index: int = 0,
) -> dict[str, Any]:
    """Train B2 on one stored real expert observation/action chunk."""

    if config.tmd.inner_source_mode != "gaussian_tm":
        raise ValueError("B2 requires tmd.inner_source_mode=gaussian_tm")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_resolved_config(config, output / "resolved_config.yaml")
    seed_everything(config.training.seed)
    record, arrays = _read_record(Path(expert_manifest), record_index)
    device = torch.device(config.training.device)
    policy, preprocessor, _ = load_smolvla_tmd(
        config.checkpoints.student_id,
        revision=config.checkpoints.student_revision,
        processor_revision=config.checkpoints.student_processor_revision,
        lerobot_commit=config.checkpoints.lerobot_commit,
        device=config.training.device,
        outer_steps=config.tmd.outer_steps,
        inner_steps=config.tmd.inner_steps,
        recurrent_layers=config.tmd.recurrent_layers,
        hidden_dim=config.tmd.hidden_dim,
        main_loss_weight=config.tmd.main_loss_weight,
        inner_source_mode=config.tmd.inner_source_mode,
    )
    policy.configure_trainable(
        train_main_action_projections=config.tmd.train_main_action_projections
    )
    trainable = {name: parameter for name, parameter in policy.named_parameters() if parameter.requires_grad}
    if not trainable or any("vision" in name.lower() for name in trainable):
        raise RuntimeError(f"unexpected trainable parameter policy: {sorted(trainable)}")
    optimizer = make_optimizer(config, trainable.values())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(config.training.expert_steps, 1)
    )
    canonical_batch = _expert_batch(record, arrays, device, config.training.batch_size)
    processed = preprocessor(canonical_batch)
    losses: list[float] = []
    started = time.perf_counter()
    policy.train()
    internal_actions = policy.base_policy.prepare_action(processed)
    diagnostic_generator = torch.Generator(device=device).manual_seed(config.training.seed + 99)
    diagnostic_outer_noise = torch.randn(
        internal_actions.shape,
        device=device,
        dtype=internal_actions.dtype,
        generator=diagnostic_generator,
    )
    diagnostic_inner_noise = torch.randn(
        internal_actions.shape,
        device=device,
        dtype=internal_actions.dtype,
        generator=diagnostic_generator,
    )
    diagnostic_time = torch.full(
        (config.training.batch_size,), 0.5, device=device, dtype=torch.float32
    )
    with torch.no_grad():
        diagnostic_initial = float(
            policy.transition_matching_loss(
                processed,
                noise=diagnostic_outer_noise,
                inner_noise=diagnostic_inner_noise,
                outer_time=diagnostic_time,
            )["loss"]
        )
    for _ in range(config.training.expert_steps):
        optimizer.zero_grad(set_to_none=True)
        result = policy.transition_matching_loss(processed)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(trainable.values(), 5.0)
        optimizer.step()
        scheduler.step()
        losses.append(float(result["loss"].detach()))
    with torch.no_grad():
        diagnostic_final = float(
            policy.transition_matching_loss(
                processed,
                noise=diagnostic_outer_noise,
                inner_noise=diagnostic_inner_noise,
                outer_time=diagnostic_time,
            )["loss"]
        )
    elapsed = time.perf_counter() - started
    metadata = {
        "base_checkpoint": config.checkpoints.student_id,
        "base_revision": config.checkpoints.student_revision,
        "teacher_checkpoint": config.checkpoints.teacher_id,
        "teacher_revision": config.checkpoints.teacher_revision,
        "lerobot_commit": config.checkpoints.lerobot_commit,
        "dataset_revision": config.dataset.revision,
        "processor_metadata": {
            "student_revision": config.checkpoints.student_processor_revision,
            "teacher_revision": config.checkpoints.teacher_processor_revision,
        },
        "training_round": 0,
        "policy_version": f"B2-expert-step-{config.training.expert_steps}",
        "replay_manifest_cursor": 0,
        "resolved_config": config.to_dict(),
        "outer_steps": config.tmd.outer_steps,
        "inner_steps": config.tmd.inner_steps,
        "inner_source_mode": config.tmd.inner_source_mode,
        "architecture": {
            "hidden_dim": config.tmd.hidden_dim,
            "recurrent_layers": config.tmd.recurrent_layers,
            "prediction_horizon": config.horizons.prediction_horizon,
            "canonical_action_dim": config.canonical.action_dim,
        },
    }
    checkpoint = output / "checkpoint.pt"
    save_training_checkpoint(
        checkpoint,
        policy=policy,
        discriminator=None,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        metadata=metadata,
    )
    report = {
        "arm": "B2",
        "data_label": "real LIBERO expert chunk",
        "sample_id": record["sample_id"],
        "task_index": record["task_index"],
        "instruction": record["instruction"],
        "training_steps": config.training.expert_steps,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "fixed_diagnostic_initial_loss": diagnostic_initial,
        "fixed_diagnostic_final_loss": diagnostic_final,
        "loss_decreased": diagnostic_final < diagnostic_initial,
        "wall_time_s": elapsed,
        "checkpoint": str(checkpoint.resolve()),
        "trainable_parameters": sorted(trainable),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable.values()),
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "training_losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "stochastic_training_loss"))
        writer.writerows(enumerate(losses))
    return report


__all__ = ["make_optimizer", "seed_everything", "train_expert_chunk"]
