"""Stage-2 DMD2-v refinement initialized from an immutable Stage-1 checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tmd_policy.backends.action_coordinates import ActionCoordinateBridge
from tmd_policy.backends.lerobot.pi05_teacher import LeRobotPI05Teacher
from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent
from tmd_policy.methods.dmd2_flow.program import DMD2FlowProgram

from .program import TMDStage1Program, sample_stage1_generator


class TMDStage2Program(DMD2FlowProgram):
    """Actual Stage-1 sampler plus online PI0.5 DMD2-v updates."""

    def __init__(
        self,
        *,
        stage1: TMDStage1Program,
        teacher: LeRobotPI05Teacher,
        bridge: ActionCoordinateBridge,
        config: dict[str, Any],
        stage1_checkpoint: str | Path,
        stage1_sha256: str,
        fake_score: nn.Module | None = None,
    ) -> None:
        student = stage1.student
        super().__init__(
            student=student,
            teacher=teacher,
            bridge=bridge,
            config=config,
            fake_score=fake_score,
        )
        self.stage1_head = stage1.head
        self.stage1_head.requires_grad_(True)
        self.stage1_config = dict(stage1.tmd_config)
        self.stage1_checkpoint = str(Path(stage1_checkpoint).resolve())
        self.stage1_sha256 = stage1_sha256
        self.outer_steps = int(config["stage1_outer_steps"])
        self.inner_steps = int(config["stage1_inner_steps"])

    def _sample_student(self, batch: dict[str, Any], *, requires_grad: bool) -> Tensor:
        processed = self.student.preprocess_observation(batch)
        condition = self.student.encode_condition(processed)
        noise = torch.randn(condition.batch_size, 50, 32, device=self.student_device, dtype=torch.float32)

        def generate() -> Tensor:
            return sample_stage1_generator(
                self.student,
                self.stage1_head,
                condition,
                noise,
                outer_steps=self.outer_steps,
                inner_steps=self.inner_steps,
            )

        if requires_grad:
            return generate()
        with torch.no_grad():
            return generate().detach()

    def make_optimizers(self, training: dict[str, Any]) -> dict[str, torch.optim.Optimizer]:
        optimizers = super().make_optimizers(training)
        parameters = []
        seen: set[int] = set()
        for module in (self.student, self.stage1_head):
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        optimizers["generator"] = torch.optim.AdamW(
            parameters,
            lr=float(self.dmd_config["generator_learning_rate"]),
            betas=tuple(training.get("betas", [0.9, 0.95])),
            weight_decay=float(training["weight_decay"]),
        )
        return optimizers

    def extra_provenance(self) -> dict[str, Any]:
        value = super().extra_provenance()
        value.update(
            {
                "stage1_checkpoint": self.stage1_checkpoint,
                "stage1_checkpoint_sha256": self.stage1_sha256,
                "stage1_config": self.stage1_config,
                "sampler_identity": "sample_stage1_generator shared by Stage 1, Stage 2, and evaluation",
            }
        )
        return value


__all__ = ["TMDStage2Program"]
