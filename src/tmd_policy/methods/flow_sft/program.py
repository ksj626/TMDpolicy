"""Executable Flow-SFT program using the official SmolVLA objective."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent
from tmd_policy.training.engine import TrainingProgram


class FlowSFTProgram(TrainingProgram):
    def __init__(self, student: LeRobotSmolVLAStudent, *, fine_tuning: dict[str, Any]) -> None:
        super().__init__()
        self.student = student
        self.fine_tuning = dict(fine_tuning)
        self.selected_names = student.configure_trainable(
            fine_tuning["mode"],
            lora_rank=int(fine_tuning.get("lora_rank", 16)),
            lora_alpha=int(fine_tuning.get("lora_alpha", 16)),
            lora_dropout=float(fine_tuning.get("lora_dropout", 0.0)),
        )

    def phase_schedule(self) -> tuple[str, ...]:
        return ("generator",)

    def loss(self, batch: dict[str, Any], phase: str) -> tuple[Tensor, dict[str, float]]:
        if phase != "generator":
            raise ValueError(f"Flow-SFT has no phase {phase!r}")
        loss = self.student.flow_matching_loss(batch)
        return loss, {"flow_matching": float(loss.detach())}

    def make_optimizers(self, training: dict[str, Any]) -> dict[str, torch.optim.Optimizer]:
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        return {
            "generator": torch.optim.AdamW(
                parameters,
                lr=float(training["learning_rate"]),
                betas=tuple(training.get("betas", [0.9, 0.95])),
                eps=float(training.get("epsilon", 1e-8)),
                weight_decay=float(training["weight_decay"]),
            )
        }

    def extra_provenance(self) -> dict[str, Any]:
        return {
            "objective": "LeRobot SmolVLAPolicy.forward flow matching",
            "fine_tuning": self.fine_tuning,
            "selected_parameter_names": list(self.selected_names),
        }


__all__ = ["FlowSFTProgram"]
