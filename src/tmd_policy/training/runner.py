from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

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


class _ExpertManifestDataset(Dataset[tuple[dict[str, Any], dict[str, np.ndarray]]]):
    def __init__(self, manifest: Path, split: str) -> None:
        self.manifest = manifest
        rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        self.rows = [row for row in rows if row.get("split", "train") == split]
        if not self.rows:
            raise ValueError(f"expert manifest contains no {split!r} records: {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        row = self.rows[index]
        with np.load(self.manifest.parent / row["payload"], allow_pickle=False) as payload:
            arrays = {key: payload[key] for key in payload.files}
        return row, arrays


def _expert_batch(
    record: dict[str, Any] | list[dict[str, Any]],
    arrays: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if isinstance(record, list):
        records = record
        array_rows = arrays
        assert isinstance(array_rows, list)
        batch = {
            "observation.state": torch.from_numpy(
                np.stack([item["path_states"][0] for item in array_rows])
            ).to(device),
            "action": torch.from_numpy(np.stack([item["plan_actions"] for item in array_rows])).to(device),
            "action_is_pad": torch.from_numpy(
                np.stack([~item["plan_valid"].astype(bool) for item in array_rows])
            ).to(device),
            "task": [item["instruction"] for item in records],
        }
        image_keys = {key for key in array_rows[0] if key.startswith("image::")}
        if any({key for key in item if key.startswith("image::")} != image_keys for item in array_rows):
            raise ValueError("expert records have inconsistent image keys")
        for key in image_keys:
            batch[key.removeprefix("image::")] = torch.from_numpy(
                np.stack([item[key] for item in array_rows])
            ).to(device)
        return batch
    assert isinstance(arrays, dict)
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
    record_index: int | None = None,
) -> dict[str, Any]:
    """Train B2 over every training record through a seeded DataLoader."""

    if config.tmd.inner_source_mode != "gaussian_tm":
        raise ValueError("B2 requires tmd.inner_source_mode=gaussian_tm")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_resolved_config(config, output / "resolved_config.yaml")
    seed_everything(config.training.seed)
    if record_index is not None:
        raise ValueError("single-record training was retired; omit record_index to use the train split")
    manifest_path = Path(expert_manifest)
    train_dataset = _ExpertManifestDataset(manifest_path, "train")
    validation_dataset = _ExpertManifestDataset(manifest_path, "validation")
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
        dropout=config.tmd.dropout,
        main_loss_weight=config.tmd.main_loss_weight,
        transition_loss=config.tmd.loss,
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
    def collate(rows: list[tuple[dict[str, Any], dict[str, np.ndarray]]]) -> dict[str, Any]:
        result = _expert_batch([row[0] for row in rows], [row[1] for row in rows], device, len(rows))
        result["metadata"] = [row[0] for row in rows]
        return result

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.training.seed),
        num_workers=0,
        collate_fn=collate,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=config.training.batch_size, collate_fn=collate)
    fixed_validation_batch = next(iter(validation_loader))
    fixed_validation_batch.pop("metadata")
    fixed_validation = preprocessor(fixed_validation_batch)
    losses: list[float] = []
    started = time.perf_counter()
    policy.train()
    internal_actions = policy.base_policy.prepare_action(fixed_validation)
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
        (internal_actions.shape[0],), 0.5, device=device, dtype=torch.float32
    )
    with torch.no_grad():
        diagnostic_initial = float(
            policy.transition_matching_loss(
                fixed_validation,
                noise=diagnostic_outer_noise,
                inner_noise=diagnostic_inner_noise,
                outer_time=diagnostic_time,
            )["loss"]
        )
    train_iterator = iter(train_loader)
    records_seen: set[str] = set()
    for _ in range(config.training.expert_steps):
        try:
            canonical_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            canonical_batch = next(train_iterator)
        records_seen.update(item["sample_id"] for item in canonical_batch.pop("metadata", []))
        processed = preprocessor(canonical_batch)
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
                fixed_validation,
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
        "train_main_action_projections": config.tmd.train_main_action_projections,
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
        "training_records": len(train_dataset),
        "validation_records": len(validation_dataset),
        "records_seen": sorted(records_seen),
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
