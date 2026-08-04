"""Construct real assets and model graphs for every retained training command."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from tmd_policy.backends.action_coordinates import ActionCoordinateBridge, ActionNormalizer
from tmd_policy.backends.lerobot.pi05_teacher import LeRobotPI05Teacher
from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent
from tmd_policy.config import project_path
from tmd_policy.data.libero import LeRobotLiberoChunks
from tmd_policy.data.occupancy import (
    BalancedOccupancyDataset,
    ExpertOccupancyWindows,
    StudentOccupancyWindows,
)
from tmd_policy.methods.dmd2_flow import DMD2FlowProgram, PI05CloneFakeScore, SmolVLACloneFakeScore
from tmd_policy.methods.flow_sft import FlowSFTProgram
from tmd_policy.methods.occupancy_tmd import (
    OccupancyDiscriminator,
    OccupancyDiscriminatorProgram,
    OccupancyWeightedTMDProgram,
    WindowNormalizer,
)
from tmd_policy.methods.tmd import TMDStage1Program, TMDStage2Program
from tmd_policy.training.engine import TrainingProgram


@dataclass(frozen=True)
class TrainingBundle:
    program: TrainingProgram
    train_dataset: Dataset
    validation_dataset: Dataset


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cache(config: dict[str, Any]) -> Path:
    return project_path(config["dataset"]["cache"])


def build_student(config: dict[str, Any], *, device: str | None = None) -> LeRobotSmolVLAStudent:
    asset = config["models"]["student"]
    return LeRobotSmolVLAStudent.from_pretrained(
        asset["id"],
        revision=asset["revision"],
        processor_revision=asset["processor_revision"],
        device=device or config["training"]["device"],
        cache_dir=_cache(config) / "hub",
        local_files_only=bool(config["backend"].get("local_files_only", False)),
        expected_source_hashes=config["backend"].get("expected_source_hashes"),
    )


def build_teacher(config: dict[str, Any], *, device: str | None = None) -> LeRobotPI05Teacher:
    asset = config["models"]["teacher"]
    dmd = config.get("dmd2", config.get("stage2", {}))
    return LeRobotPI05Teacher.from_pretrained(
        asset["id"],
        revision=asset["revision"],
        processor_revision=asset["processor_revision"],
        device=device or dmd.get("teacher_device", config["training"]["device"]),
        dtype=dmd.get("teacher_dtype", "bfloat16"),
        minimum_score_time=float(dmd.get("minimum_score_time", 1e-3)),
        cache_dir=_cache(config) / "hub",
        local_files_only=bool(config["backend"].get("local_files_only", False)),
        expected_source_hashes=config["backend"].get("expected_source_hashes"),
    )


def build_expert_datasets(config: dict[str, Any]) -> tuple[LeRobotLiberoChunks, LeRobotLiberoChunks]:
    arguments = {
        "manifest_path": project_path(config["dataset"]["manifest"]),
        "root": _cache(config) / "datasets" / "lerobot--libero",
        "download_videos": bool(config["dataset"].get("download_videos", True)),
        "video_backend": config["dataset"].get("video_backend"),
    }
    return LeRobotLiberoChunks(split="train", **arguments), LeRobotLiberoChunks(
        split="validation", **arguments
    )


def _fake_score(
    config: dict[str, Any],
    dmd: dict[str, Any],
) -> torch.nn.Module | None:
    variant = dmd["fake_score_variant"]
    if variant == "lightweight":
        return None
    fake_device = dmd.get("fake_score_device", config["training"]["device"])
    if variant == "smolvla_clone":
        return SmolVLACloneFakeScore(build_student(config, device=fake_device))
    if variant == "pi05_clone":
        return PI05CloneFakeScore(build_teacher(config, device=fake_device))
    raise ValueError(f"unknown fake-score variant: {variant}")


def _load_payload(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    resolved = project_path(path)
    actual = file_sha256(resolved)
    if actual != expected_sha256:
        raise RuntimeError(f"immutable checkpoint SHA-256 mismatch for {resolved}: {actual} != {expected_sha256}")
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if payload.get("format") != "tmdpolicy.training/v1":
        raise ValueError(f"unsupported TMDpolicy checkpoint: {resolved}")
    return payload


def _occupancy_network(config: dict[str, Any]) -> OccupancyDiscriminator:
    value = config["occupancy_model"]
    return OccupancyDiscriminator(
        num_tasks=int(value["num_tasks"]),
        window_length=int(value["window_length"]),
        model_dim=int(value["model_dim"]),
        layers=int(value["layers"]),
        heads=int(value["heads"]),
        feedforward_dim=int(value["feedforward_dim"]),
        dropout=float(value.get("dropout", 0.0)),
    )


def _normalizer_from_state(state: dict[str, torch.Tensor], *, prefix: str = "normalizer.") -> WindowNormalizer:
    def value(name: str) -> torch.Tensor:
        key = prefix + name
        if key not in state:
            raise ValueError(f"occupancy checkpoint missing {key}")
        return state[key]

    return WindowNormalizer(
        value("state_mean"),
        value("state_std"),
        value("action_mean"),
        value("action_std"),
        value("visual_mean"),
        value("visual_std"),
        fitted_samples=-1,
    )


def build_training_bundle(config: dict[str, Any]) -> TrainingBundle:
    method = config["method"]
    if method in {"flow_sft", "tmd_stage1", "dmd2_flow", "tmd_stage2", "occupancy_tmd"}:
        train, validation = build_expert_datasets(config)
    if method == "flow_sft":
        return TrainingBundle(
            FlowSFTProgram(build_student(config), fine_tuning=config["fine_tuning"]), train, validation
        )
    if method == "tmd_stage1":
        return TrainingBundle(TMDStage1Program(build_student(config), config["tmd"]), train, validation)
    if method == "dmd2_flow":
        student = build_student(config)
        teacher = build_teacher(config)
        bridge = ActionCoordinateBridge.from_processors(
            student.preprocessor,
            teacher.preprocessor,
            student_config=student.policy.config,
            teacher_config=teacher.policy.config,
        )
        return TrainingBundle(
            DMD2FlowProgram(
                student=student,
                teacher=teacher,
                bridge=bridge,
                config=config["dmd2"],
                fake_score=_fake_score(config, config["dmd2"]),
            ),
            train,
            validation,
        )
    if method == "tmd_stage2":
        stage2 = config["stage2"]
        student = build_student(config)
        stage1 = TMDStage1Program(student, config["stage1_architecture"])
        payload = _load_payload(stage2["stage1_checkpoint"], stage2["stage1_checkpoint_sha256"])
        if payload["config"]["method"] != "tmd_stage1":
            raise ValueError("Stage-2 initialization checkpoint was not produced by tmd-stage1")
        stage1.load_state_dict(payload["program"], strict=True)
        teacher = build_teacher(config)
        bridge = ActionCoordinateBridge.from_processors(
            student.preprocessor,
            teacher.preprocessor,
            student_config=student.policy.config,
            teacher_config=teacher.policy.config,
        )
        program = TMDStage2Program(
            stage1=stage1,
            teacher=teacher,
            bridge=bridge,
            config=stage2,
            stage1_checkpoint=project_path(stage2["stage1_checkpoint"]),
            stage1_sha256=stage2["stage1_checkpoint_sha256"],
            fake_score=_fake_score(config, stage2),
        )
        return TrainingBundle(program, train, validation)
    if method == "occupancy_discriminator":
        model_cfg = config["occupancy_model"]
        arguments = {
            "manifest_path": project_path(config["dataset"]["manifest"]),
            "root": _cache(config) / "datasets" / "lerobot--libero",
            "window_length": int(model_cfg["window_length"]),
            "download_videos": bool(config["dataset"].get("download_videos", True)),
        }
        train_expert = ExpertOccupancyWindows(split="train", **arguments)
        validation_expert = ExpertOccupancyWindows(split="validation", **arguments)
        rollout_path = project_path(config["rollouts"]["store"])
        train_student = StudentOccupancyWindows(
            rollout_path,
            "train",
            window_length=int(model_cfg["window_length"]),
            stride=int(config["rollouts"].get("window_stride", 1)),
        )
        validation_student = StudentOccupancyWindows(
            rollout_path,
            "validation",
            window_length=int(model_cfg["window_length"]),
            stride=int(config["rollouts"].get("window_stride", 1)),
        )
        train_dataset = BalancedOccupancyDataset(train_expert, train_student)
        validation_dataset = BalancedOccupancyDataset(validation_expert, validation_student)
        normalizer = WindowNormalizer.fit(
            train_dataset, max_samples=int(model_cfg["normalization_max_windows"])
        )
        return TrainingBundle(
            OccupancyDiscriminatorProgram(_occupancy_network(config), normalizer),
            train_dataset,
            validation_dataset,
        )
    if method == "occupancy_tmd":
        occupancy = config["occupancy"]
        payload = _load_payload(
            occupancy["discriminator_checkpoint"], occupancy["discriminator_checkpoint_sha256"]
        )
        if payload["config"]["method"] != "occupancy_discriminator":
            raise ValueError("occupancy TMD requires an occupancy-discriminator checkpoint")
        discriminator = _occupancy_network(config)
        discriminator.load_state_dict(
            {key.removeprefix("discriminator."): value for key, value in payload["program"].items() if key.startswith("discriminator.")},
            strict=True,
        )
        normalizer = _normalizer_from_state(payload["program"])
        student = build_student(config)
        action_normalizer = ActionNormalizer.from_pipeline(student.preprocessor)
        program = OccupancyWeightedTMDProgram(
            student,
            config["tmd"],
            occupancy_discriminator=discriminator,
            occupancy_normalizer=normalizer,
            student_action_normalizer=action_normalizer,
            occupancy_config=occupancy,
        )
        return TrainingBundle(program, train, validation)
    raise ValueError(f"no training builder for method={method!r}")


__all__ = [
    "TrainingBundle",
    "build_expert_datasets",
    "build_student",
    "build_teacher",
    "build_training_bundle",
    "file_sha256",
]
