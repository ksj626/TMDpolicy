from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tmd_policy.common.capabilities import Capability, CapabilitySet
from tmd_policy.common.checkpointing import load_method_checkpoint, save_method_checkpoint
from tmd_policy.common.density import DivergenceMode
from tmd_policy.methods.base import DryRunReport, ResearchMethod

from .losses import categorical_opd_loss, continuous_flow_opd_loss


@dataclass(frozen=True)
class Pi05ProbabilityCapability:
    model_revision: str
    processor_revision: str
    exposes_normalized_token_probabilities: bool = False
    exposes_conditional_action_log_density: bool = False
    exposes_supported_invertible_vector_field: bool = False

    def capability_set(self) -> CapabilitySet:
        values: list[Capability] = []
        if self.exposes_normalized_token_probabilities:
            values.append(Capability.TOKEN_LOG_PROBABILITY)
        if self.exposes_conditional_action_log_density:
            values.extend((Capability.EXACT_LOG_DENSITY, Capability.TEACHER_AT_STUDENT_ACTION))
        if self.exposes_supported_invertible_vector_field:
            values.append(Capability.FLOW_VELOCITY)
        return CapabilitySet.of(
            f"pi0.5@{self.model_revision}",
            *values,
            reason=(
                "pinned LeRobot PI05Policy exposes sampling and an internal denoise_step, "
                "but no supported normalized log_prob/inverse-flow API"
            ),
        )


@dataclass(frozen=True)
class OPDConfig:
    group_size: int = 8
    learning_rate: float = 1e-6
    density_steps: int = 32
    divergence_mode: str = "exact"
    require_fresh_policy_version: bool = True

    def __post_init__(self) -> None:
        if self.group_size < 1 or self.learning_rate <= 0 or self.density_steps < 1:
            raise ValueError("invalid OPD group/optimizer/density configuration")
        DivergenceMode(self.divergence_mode)


class OPDMethod(ResearchMethod):
    classification = "exact categorical reproduction or named continuous-flow port"

    def __init__(
        self,
        *,
        mode: str,
        student: nn.Module,
        teacher: nn.Module,
        config: OPDConfig,
        capabilities: CapabilitySet,
        current_policy_version: str,
        dry_run: DryRunReport | None = None,
    ) -> None:
        if mode not in {"opd_categorical", "continuous_flow_opd"}:
            raise ValueError("unknown OPD mode")
        self.name = mode
        required = (
            {Capability.ON_POLICY_ROLLOUTS, Capability.TOKEN_LOG_PROBABILITY}
            if mode == "opd_categorical"
            else {
                Capability.ON_POLICY_ROLLOUTS,
                Capability.EXACT_LOG_DENSITY,
                Capability.TEACHER_AT_STUDENT_ACTION,
            }
        )
        capabilities.require(required, method=mode)
        self.student, self.teacher, self.config = student, teacher, config
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(student.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        self.current_policy_version = current_policy_version
        self.round = 0
        self._dry_run = dry_run

    def required_data_capabilities(self) -> frozenset[Capability]:
        base = {Capability.ON_POLICY_ROLLOUTS}
        base.add(
            Capability.TOKEN_LOG_PROBABILITY
            if self.name == "opd_categorical"
            else Capability.EXACT_LOG_DENSITY
        )
        return frozenset(base)

    def validate_config(self) -> None:
        self.config.__post_init__()

    def _validate_freshness(self, batch: Mapping[str, Any]) -> None:
        versions = set(batch["policy_versions"])
        rounds = {int(value) for value in batch["collection_rounds"]}
        if self.config.require_fresh_policy_version and versions != {self.current_policy_version}:
            raise RuntimeError(
                f"exact OPD requires fresh current-policy rollouts: {versions} != {self.current_policy_version}"
            )
        if rounds != {self.round}:
            raise RuntimeError(f"OPD collection round mismatch: {rounds} != {self.round}")
        group_ids = tuple(batch["group_ids"])
        counts = {group: group_ids.count(group) for group in set(group_ids)}
        if not counts or set(counts.values()) != {self.config.group_size}:
            raise RuntimeError(
                f"OPD requires complete on-policy groups of size {self.config.group_size}: {counts}"
            )

    def training_step(self, batch: Mapping[str, Any]) -> Mapping[str, Tensor]:
        self._validate_freshness(batch)
        if self.name == "opd_categorical":
            result = categorical_opd_loss(
                batch["student_logits"],
                batch["teacher_logits"],
                batch["sampled_tokens"],
                batch["valid_mask"],
            )
        else:
            result = continuous_flow_opd_loss(
                batch["actions"],
                student_vector_field=batch["student_vector_field"],
                teacher_vector_field=batch["teacher_vector_field"],
                integration_steps=self.config.density_steps,
                divergence_mode=self.config.divergence_mode,
                student_probe=batch.get("student_probe"),
                teacher_probe=batch.get("teacher_probe"),
            )
        return result

    def advance_round(self, new_policy_version: str) -> None:
        if not new_policy_version or new_policy_version == self.current_policy_version:
            raise ValueError("a new OPD round needs a new policy version")
        self.round += 1
        self.current_policy_version = new_policy_version

    def sample_action_chunk(self, batch: Mapping[str, Any]) -> Tensor:
        return self.student.sample_action_chunk(batch)

    def save_method_state(self, path: str | Path) -> Path:
        return save_method_checkpoint(
            path,
            method_name=self.name,
            models={"student": self.student, "teacher": self.teacher},
            optimizers={"student": self.optimizer},
            schedulers={"student": self.scheduler},
            scaler=None,
            counters={"round": self.round},
            config={**asdict(self.config), "current_policy_version": self.current_policy_version},
            provenance={},
            trainable_names={
                "student": [n for n, p in self.student.named_parameters() if p.requires_grad],
                "teacher": [],
            },
        )

    def load_method_state(self, path: str | Path) -> Mapping[str, Any]:
        value = load_method_checkpoint(
            path,
            expected_method=self.name,
            models={"student": self.student, "teacher": self.teacher},
            optimizers={"student": self.optimizer},
            schedulers={"student": self.scheduler},
        )
        self.round = value["counters"]["round"]
        self.current_policy_version = value["config"]["current_policy_version"]
        return value

    def dry_run_report(self) -> DryRunReport:
        if self._dry_run is None:
            raise RuntimeError("dry-run context was not provided")
        return self._dry_run
