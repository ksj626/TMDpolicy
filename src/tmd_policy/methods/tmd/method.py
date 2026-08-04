from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tmd_policy.common.capabilities import Capability, CapabilitySet
from tmd_policy.common.checkpointing import load_method_checkpoint, save_method_checkpoint
from tmd_policy.methods.base import DryRunReport, ResearchMethod

from .meanflow import ActionMeanFlowHead, MeanFlowConfig, inner_flow_rollout, meanflow_loss


class TMDMethod(ResearchMethod):
    name = "tmd"
    classification = "paper-faithful action-space port"

    def __init__(
        self,
        *,
        backbone: nn.Module,
        head: ActionMeanFlowHead,
        config: MeanFlowConfig,
        stage: int,
        capabilities: CapabilitySet,
        dry_run: DryRunReport | None = None,
    ) -> None:
        if stage not in {1, 2}:
            raise ValueError("TMD stage must be 1 or 2")
        self.backbone, self.head, self.config, self.stage = backbone, head, config, stage
        required = {Capability.EXPERT_ACTION_CHUNKS, Capability.FLOW_VELOCITY}
        if stage == 2:
            required |= {Capability.FLOW_SCORE, Capability.TEACHER_AT_STUDENT_ACTION}
        capabilities.require(required, method=f"TMD Stage {stage}")
        self.optimizer = torch.optim.AdamW(
            list(head.parameters()) + [p for p in backbone.parameters() if p.requires_grad], lr=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        self.step = 0
        self._dry_run = dry_run

    def required_data_capabilities(self) -> frozenset[Capability]:
        values = {Capability.EXPERT_ACTION_CHUNKS, Capability.FLOW_VELOCITY}
        if self.stage == 2:
            values |= {Capability.FLOW_SCORE, Capability.TEACHER_AT_STUDENT_ACTION}
        return frozenset(values)

    def validate_config(self) -> None:
        self.config.__post_init__()

    def training_step(self, batch: Mapping[str, Any]) -> Mapping[str, Tensor]:
        if self.stage == 2:
            raise RuntimeError("TMD Stage 2 must be constructed with the DMD2-v coordinator")
        outer_time = batch["outer_time"]
        expanded_time = outer_time
        while expanded_time.ndim < batch["actions"].ndim:
            expanded_time = expanded_time.unsqueeze(-1)
        outer_state = (
            (1 - expanded_time) * batch["actions"] + expanded_time * batch["outer_source"]
        )
        features = self.backbone(outer_state, outer_time)
        result = meanflow_loss(
            self.head,
            outer_data=batch["actions"],
            outer_source=batch["outer_source"],
            outer_time=batch["outer_time"],
            inner_source=batch["inner_source"],
            inner_time=batch["inner_time"],
            target_time=batch["target_time"],
            features=features,
            valid_mask=batch["valid_mask"],
            config=self.config,
        )
        self.step += 1
        return result

    def sample_action_chunk(self, batch: Mapping[str, Any]) -> Tensor:
        features = self.backbone(batch["outer_state"], batch["outer_time"])
        transition = inner_flow_rollout(
            self.head,
            inner_source=batch["inner_source"],
            features=features,
            time_grid=batch["inner_time_grid"],
        )
        return batch["outer_state"] - batch["outer_step"] * transition

    def save_method_state(self, path: str | Path) -> Path:
        return save_method_checkpoint(
            path,
            method_name=f"{self.name}_stage{self.stage}",
            models={"backbone": self.backbone, "flow_head": self.head},
            optimizers={"student": self.optimizer},
            schedulers={"student": self.scheduler},
            scaler=None,
            counters={"step": self.step, "stage": self.stage},
            config=asdict(self.config),
            provenance={},
            trainable_names={
                "backbone": [n for n, p in self.backbone.named_parameters() if p.requires_grad],
                "flow_head": [n for n, p in self.head.named_parameters() if p.requires_grad],
            },
        )

    def load_method_state(self, path: str | Path) -> Mapping[str, Any]:
        value = load_method_checkpoint(
            path,
            expected_method=f"{self.name}_stage{self.stage}",
            models={"backbone": self.backbone, "flow_head": self.head},
            optimizers={"student": self.optimizer},
            schedulers={"student": self.scheduler},
        )
        self.step = value["counters"]["step"]
        return value

    def dry_run_report(self) -> DryRunReport:
        if self._dry_run is None:
            raise RuntimeError("dry-run context was not provided")
        return self._dry_run


class TMDStage2Method(ResearchMethod):
    """Stage-2 coordinator whose generator is the initialized TMD inner flow."""

    name = "tmd_stage2"
    classification = "paper-faithful action-space port with DMD2-v"

    def __init__(
        self,
        *,
        dmd2_v: ResearchMethod,
        stage1_checkpoint_hash: str,
        capabilities: CapabilitySet,
        dry_run: DryRunReport | None = None,
    ) -> None:
        capabilities.require(
            {
                Capability.EXPERT_ACTION_CHUNKS,
                Capability.FLOW_SCORE,
                Capability.TEACHER_AT_STUDENT_ACTION,
            },
            method=self.name,
        )
        if not stage1_checkpoint_hash:
            raise ValueError("TMD Stage 2 requires immutable Stage-1 checkpoint identity")
        for attribute in (
            "generator",
            "fake_score",
            "discriminator",
            "optimizers",
            "schedulers",
            "counters",
        ):
            if not hasattr(dmd2_v, attribute):
                raise TypeError(f"DMD2-v coordinator lacks {attribute}")
        self.dmd2_v = dmd2_v
        self.stage1_checkpoint_hash = stage1_checkpoint_hash
        self._dry_run = dry_run

    def required_data_capabilities(self) -> frozenset[Capability]:
        return frozenset(
            {
                Capability.EXPERT_ACTION_CHUNKS,
                Capability.FLOW_SCORE,
                Capability.TEACHER_AT_STUDENT_ACTION,
            }
        )

    def validate_config(self) -> None:
        self.dmd2_v.validate_config()

    def training_step(self, batch: Mapping[str, Any]) -> Mapping[str, Tensor]:
        return self.dmd2_v.training_step(batch)

    def sample_action_chunk(self, batch: Mapping[str, Any]) -> Tensor:
        return self.dmd2_v.sample_action_chunk(batch)

    def save_method_state(self, path: str | Path) -> Path:
        coordinator = self.dmd2_v
        return save_method_checkpoint(
            path,
            method_name=self.name,
            models={
                "tmd_generator": coordinator.generator,
                "fake_score": coordinator.fake_score,
                "dmd2_gan": coordinator.discriminator,
            },
            optimizers=coordinator.optimizers,
            schedulers=coordinator.schedulers,
            scaler=None,
            counters=coordinator.counters,
            config={
                "stage1_checkpoint_hash": self.stage1_checkpoint_hash,
                "dmd2_v": asdict(coordinator.config),
            },
            provenance={},
            trainable_names={
                "tmd_generator": [name for name, _ in coordinator.generator.named_parameters()],
                "fake_score": [name for name, _ in coordinator.fake_score.named_parameters()],
                "dmd2_gan": [name for name, _ in coordinator.discriminator.named_parameters()],
            },
        )

    def load_method_state(self, path: str | Path) -> Mapping[str, Any]:
        coordinator = self.dmd2_v
        value = load_method_checkpoint(
            path,
            expected_method=self.name,
            models={
                "tmd_generator": coordinator.generator,
                "fake_score": coordinator.fake_score,
                "dmd2_gan": coordinator.discriminator,
            },
            optimizers=coordinator.optimizers,
            schedulers=coordinator.schedulers,
        )
        if value["config"]["stage1_checkpoint_hash"] != self.stage1_checkpoint_hash:
            raise RuntimeError("TMD Stage-1 checkpoint identity changed across resume")
        coordinator.counters = dict(value["counters"])
        return value

    def dry_run_report(self) -> DryRunReport:
        if self._dry_run is None:
            raise RuntimeError("dry-run context was not provided")
        return self._dry_run
