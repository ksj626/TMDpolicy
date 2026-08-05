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
from tmd_policy.methods.flow_objectives import shifted_time_grid

from .program import TMDStage1Program
from .meanflow import integrate_inner_flow


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
        intended_before = tuple(stage1.intended_student_trainable_names)
        super().__init__(
            student=student,
            teacher=teacher,
            bridge=bridge,
            config=config,
            fake_score=fake_score,
            preserve_student_trainability=True,
        )
        self.stage1_head = stage1.head
        self.stage1_head.requires_grad_(True)
        self.stage1_config = dict(stage1.tmd_config)
        self.stage1_checkpoint = str(Path(stage1_checkpoint).resolve())
        self.stage1_sha256 = stage1_sha256
        self.outer_steps = int(self.stage1_config["discrete_outer_steps"])
        self.inner_steps = int(self.stage1_config["discrete_inner_steps"])
        self.student_time_shift_gamma = float(
            self.stage1_config["student_time_shift_gamma"]
        )
        intended_after = tuple(student.trainable_parameter_names)
        if intended_after != intended_before:
            raise RuntimeError("Stage-2 construction changed the Stage-1 TMD trainability set")
        self.intended_student_trainable_names = intended_after

    def validate_phase_gradients(self, phase: str) -> None:
        if phase != "generator":
            return
        # Names recorded by configure_tmd_trainable() are relative to
        # student.policy, so validate against that same namespace.
        parameters = dict(self.student.policy.named_parameters())
        expert_names = [
            name
            for name in self.intended_student_trainable_names
            if ".lm_expert.layers." in name
        ]
        if not expert_names:
            raise RuntimeError("TMD Stage 2 has no intended trainable final expert-block parameters")
        if not any(
            parameters[name].grad is not None
            and torch.isfinite(parameters[name].grad).all()
            and parameters[name].grad.abs().sum() > 0
            for name in expert_names
        ):
            raise RuntimeError(
                "TMD Stage-2 generator produced no nonzero finite gradient in an intended "
                "final SmolVLA expert block"
            )

    def _sample_student(self, batch: dict[str, Any], *, requires_grad: bool) -> Tensor:
        processed = self.student.preprocess_observation(batch)
        condition = self.student.encode_condition(processed)
        clean = self.student.policy.prepare_action(processed)
        outer_noise = torch.randn_like(clean)
        grid = shifted_time_grid(
            self.outer_steps,
            self.student_time_shift_gamma,
            device=self.student_device,
            dtype=torch.float32,
        )
        indices = torch.randint(1, self.outer_steps + 1, (clean.shape[0],), device=self.student_device)
        outer_time = grid[indices]
        x_t = (1.0 - outer_time[:, None, None]) * clean + outer_time[:, None, None] * outer_noise
        inner_source = torch.randn_like(clean)

        def generate() -> Tensor:
            base_velocity, features = self.student.velocity_with_features(
                condition, x_t, outer_time
            )
            transition = integrate_inner_flow(
                self.stage1_head,
                inner_source,
                features,
                num_steps=self.inner_steps,
                student_time_shift_gamma=self.student_time_shift_gamma,
                base_velocity=base_velocity,
            )
            # TMD Eq. (rollout): x_hat = x_1 - predicted transition.
            return outer_noise - transition

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
                "sampler_identity": (
                    "Stage-1 checkpoint shifted outer/inner grids shared by Stage-2 outer-transition "
                    "training and sample_stage1_generator evaluation"
                ),
                "training_outer_transition": (
                    "real x, independent x1, discrete shifted t_i, x_t=(1-t_i)x+t_i*x1; "
                    "all inner steps differentiable"
                ),
                "intended_trainable_parameter_names": list(self.intended_student_trainable_names),
            }
        )
        return value


__all__ = ["TMDStage2Program"]
