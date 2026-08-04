"""Construct real assets and model graphs for every retained training command."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from tmd_policy.backends.action_coordinates import ActionCoordinateBridge
from tmd_policy.backends.lerobot.pi05_teacher import LeRobotPI05Teacher
from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent
from tmd_policy.config import project_path
from tmd_policy.data.libero import LeRobotLiberoChunks
from tmd_policy.data.occupancy import (
    BalancedOccupancyDataset,
    DeterministicStratifiedBatchSampler,
    ExpertOccupancyWindows,
    StudentOccupancyWindows,
)
from tmd_policy.methods.dmd2_flow import DMD2FlowProgram, PI05CloneFakeScore, SmolVLACloneFakeScore
from tmd_policy.methods.flow_sft import FlowSFTProgram
from tmd_policy.methods.occupancy_tmd import (
    OccupancyWeightedTMDProgram,
    ReplanOccupancyDiscriminatorProgram,
)
from tmd_policy.methods.tmd import TMDStage1Program, TMDStage2Program
from tmd_policy.training.engine import TrainingProgram


@dataclass(frozen=True)
class TrainingBundle:
    program: TrainingProgram
    train_dataset: Dataset
    validation_dataset: Dataset
    train_batch_sampler: Any | None = None


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
    *,
    teacher: LeRobotPI05Teacher,
) -> torch.nn.Module | None:
    variant = dmd["fake_score_variant"]
    if variant == "lightweight":
        return None
    fake_device = dmd.get("fake_score_device", config["training"]["device"])
    if variant == "smolvla_clone":
        return SmolVLACloneFakeScore(build_student(config, device=fake_device))
    if variant == "pi05_clone":
        if torch.device(fake_device) != teacher.device:
            raise RuntimeError(
                "PI0.5 fake-score suffix shares the immutable teacher prefix cache and must use "
                f"the teacher device ({teacher.device}); got fake_score_device={fake_device}"
            )
        clone = PI05CloneFakeScore(teacher).to(teacher.device)
        clone.verify_initialization()
        return clone
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
                fake_score=_fake_score(config, config["dmd2"], teacher=teacher),
            ),
            train,
            validation,
        )
    if method == "tmd_stage2":
        stage2 = config["stage2"]
        student = build_student(config)
        stage1 = TMDStage1Program(student, config["stage1_architecture"])
        if str(stage2["stage1_checkpoint_sha256"]).lower() == "auto":
            stage2["stage1_checkpoint_sha256"] = file_sha256(
                project_path(stage2["stage1_checkpoint"])
            )
        payload = _load_payload(stage2["stage1_checkpoint"], stage2["stage1_checkpoint_sha256"])
        if payload["config"]["method"] != "tmd_stage1":
            raise ValueError("Stage-2 initialization checkpoint was not produced by tmd-stage1")
        if payload["config"]["tmd"] != config["stage1_architecture"]:
            raise ValueError("Stage-1 architecture/training semantics differ from Stage-2 expectations")
        if payload["trainable_parameter_names"] != stage1.trainable_parameter_names():
            raise ValueError("Stage-1 checkpoint trainable identities differ from the reconstructed program")
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
            fake_score=_fake_score(config, stage2, teacher=teacher),
        )
        return TrainingBundle(program, train, validation)
    if method == "occupancy_discriminator":
        model_cfg = config["occupancy_discriminator"]
        arguments = {
            "manifest_path": project_path(config["dataset"]["manifest"]),
            "root": _cache(config) / "datasets" / "lerobot--libero",
            "window_length": int(model_cfg["window_length"]),
            "download_videos": bool(config["dataset"].get("download_videos", True)),
            "video_backend": config["dataset"].get("video_backend"),
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
        configured_tasks = config["rollouts"].get("configured_task_indices", list(range(40)))
        train_dataset = BalancedOccupancyDataset(
            train_expert, train_student, configured_tasks=configured_tasks
        )
        validation_dataset = BalancedOccupancyDataset(
            validation_expert, validation_student, configured_tasks=configured_tasks
        )
        student = build_student(config)
        teacher = build_teacher(config) if model_cfg["variant"] == "pi05_intermediate_features" else None
        bridge = (
            ActionCoordinateBridge.from_processors(
                student.preprocessor,
                teacher.preprocessor,
                student_config=student.policy.config,
                teacher_config=teacher.policy.config,
            )
            if teacher is not None
            else None
        )
        sampler = DeterministicStratifiedBatchSampler(
            train_dataset,
            int(config["training"]["batch_size"]),
            seed=int(config["training"]["seed"]),
            balance_position_bins=bool(config["rollouts"].get("balance_position_bins", False)),
        )
        return TrainingBundle(
            ReplanOccupancyDiscriminatorProgram(
                teacher=teacher,
                student=student,
                bridge=bridge,
                config=model_cfg,
            ),
            train_dataset,
            validation_dataset,
            sampler,
        )
    if method == "occupancy_tmd":
        occupancy = config["occupancy"]
        if str(occupancy["discriminator_checkpoint_sha256"]).lower() == "auto":
            occupancy["discriminator_checkpoint_sha256"] = file_sha256(
                project_path(occupancy["discriminator_checkpoint"])
            )
        payload = _load_payload(
            occupancy["discriminator_checkpoint"], occupancy["discriminator_checkpoint_sha256"]
        )
        if payload["config"]["method"] != "occupancy_discriminator":
            raise ValueError("occupancy TMD requires an occupancy-discriminator checkpoint")
        checkpoint_config = payload["config"]
        if checkpoint_config["models"] != config["models"] or checkpoint_config["dataset"] != config["dataset"]:
            raise ValueError("occupancy discriminator model/dataset provenance differs from occupancy TMD")
        discriminator_config = checkpoint_config["occupancy_discriminator"]
        variant = str(discriminator_config["variant"])
        student = build_student(config)
        teacher = build_teacher(config) if variant == "pi05_intermediate_features" else None
        bridge = (
            ActionCoordinateBridge.from_processors(
                student.preprocessor,
                teacher.preprocessor,
                student_config=student.policy.config,
                teacher_config=teacher.policy.config,
            )
            if teacher is not None
            else None
        )
        occupancy_program = ReplanOccupancyDiscriminatorProgram(
            teacher=teacher,
            student=student,
            bridge=bridge,
            config=discriminator_config,
        )
        discriminator_state = {
            key.removeprefix("discriminator."): value
            for key, value in payload["program"].items()
            if key.startswith("discriminator.")
        }
        if not discriminator_state:
            raise ValueError(
                "occupancy checkpoint predates the replan-record discriminator; recollect/retrain v2"
            )
        occupancy_program.discriminator.load_state_dict(discriminator_state, strict=True)
        scoring_config = {**discriminator_config, **occupancy}
        program = OccupancyWeightedTMDProgram(
            student,
            config["tmd"],
            occupancy_discriminator=occupancy_program.discriminator,
            occupancy_teacher=teacher,
            occupancy_bridge=bridge,
            occupancy_variant=variant,
            occupancy_config=scoring_config,
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
