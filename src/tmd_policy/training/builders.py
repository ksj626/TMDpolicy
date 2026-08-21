"""Construct the single supported DMD2 training graph and its immutable assets."""

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
from tmd_policy.methods.dmd2_flow import DMD2FlowProgram, PI05CloneFakeScore
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
    default_device = (
        config["training"]["device"] if "training" in config else config["policy"]["device"]
    )
    return LeRobotSmolVLAStudent.from_pretrained(
        asset["id"],
        revision=asset["revision"],
        processor_revision=asset["processor_revision"],
        device=device or default_device,
        cache_dir=_cache(config) / "hub",
        local_files_only=bool(config["backend"].get("local_files_only", False)),
        expected_source_hashes=config["backend"].get("expected_source_hashes"),
    )


def build_teacher(config: dict[str, Any], *, device: str | None = None) -> LeRobotPI05Teacher:
    asset = config["models"]["teacher"]
    dmd2 = config.get("dmd2", {})
    default_device = dmd2.get("teacher_device", config.get("policy", {}).get("device", "cuda:0"))
    return LeRobotPI05Teacher.from_pretrained(
        asset["id"],
        revision=asset["revision"],
        processor_revision=asset["processor_revision"],
        device=device or default_device,
        dtype=dmd2.get("teacher_dtype", "bfloat16"),
        minimum_score_time=float(dmd2.get("minimum_score_time", 1e-3)),
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
    return (
        LeRobotLiberoChunks(split="train", **arguments),
        LeRobotLiberoChunks(split="validation", **arguments),
    )


def build_training_bundle(config: dict[str, Any]) -> TrainingBundle:
    if config["method"] != "dmd2_flow":
        raise ValueError(f"unsupported training method: {config['method']!r}")

    train, validation = build_expert_datasets(config)
    student = build_student(config)
    teacher = build_teacher(config)
    dmd2 = config["dmd2"]
    if dmd2["fake_score_variant"] != "pi05_clone":
        raise ValueError("DMD2 requires fake_score_variant=pi05_clone")
    fake_device = torch.device(dmd2["fake_score_device"])
    if fake_device != teacher.device:
        raise RuntimeError(
            "PI0.5 fake-score suffix shares the immutable teacher prefix cache and must use "
            f"the teacher device ({teacher.device}); got fake_score_device={fake_device}"
        )
    fake_score = PI05CloneFakeScore(teacher).to(teacher.device)
    fake_score.verify_initialization()
    bridge = ActionCoordinateBridge.from_processors(
        student.preprocessor,
        teacher.preprocessor,
        student_config=student.policy.config,
        teacher_config=teacher.policy.config,
    )
    program = DMD2FlowProgram(
        student=student,
        teacher=teacher,
        bridge=bridge,
        config=dmd2,
        fake_score=fake_score,
    )
    return TrainingBundle(program, train, validation)


__all__ = [
    "TrainingBundle",
    "build_expert_datasets",
    "build_student",
    "build_teacher",
    "build_training_bundle",
    "file_sha256",
]
