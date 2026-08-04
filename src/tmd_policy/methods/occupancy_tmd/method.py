from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from tmd_policy.common.capabilities import Capability, CapabilitySet
from tmd_policy.common.checkpointing import load_method_checkpoint, save_method_checkpoint
from tmd_policy.methods.base import DryRunReport, ResearchMethod

from .discriminator import OccupancyWindowDiscriminator, occupancy_discriminator_loss
from .weights import MismatchPrioritizationWeight


@dataclass(frozen=True)
class OccupancyGate:
    real_held_out: bool
    source_control_deviation: float
    calibration_error: float
    saturation_fraction: float
    support_overlap: float
    effective_sample_size: float

    def require_open(self) -> None:
        failures = []
        if not self.real_held_out:
            failures.append("diagnostics are not real held-out paths")
        if self.source_control_deviation > 0.05:
            failures.append("source-only control deviates from chance")
        if self.calibration_error > 0.1:
            failures.append("calibration error exceeds 0.1")
        if self.saturation_fraction > 0.1:
            failures.append("discriminator is saturated")
        if self.support_overlap < 0.5:
            failures.append("support overlap is below 0.5")
        if self.effective_sample_size < 20:
            failures.append("effective sample size is below 20")
        if failures:
            raise RuntimeError("occupancy-TMD gate closed: " + "; ".join(failures))


@dataclass(frozen=True)
class OccupancyTMDConfig:
    minimum_weight: float = 0.5
    maximum_weight: float = 2.0
    discriminator_learning_rate: float = 1e-4

    def __post_init__(self) -> None:
        if self.minimum_weight <= 0 or self.maximum_weight < self.minimum_weight:
            raise ValueError("invalid occupancy prioritization bounds")
        if self.discriminator_learning_rate <= 0:
            raise ValueError("discriminator learning rate must be positive")


class OccupancyDiscriminatorMethod(ResearchMethod):
    """Separate balanced-BCE training method for the causal occupancy model."""

    name = "occupancy_discriminator"
    classification = "proposed diagnostic model"

    def __init__(self, *, discriminator: OccupancyWindowDiscriminator,
                 config: OccupancyTMDConfig, capabilities: CapabilitySet,
                 dry_run: DryRunReport | None = None) -> None:
        capabilities.require({Capability.PATH_WINDOWS, Capability.ON_POLICY_ROLLOUTS}, method=self.name)
        self.discriminator, self.config, self._dry_run = discriminator, config, dry_run
        self.optimizer = torch.optim.AdamW(discriminator.parameters(), lr=config.discriminator_learning_rate)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        self.step = 0

    def required_data_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.PATH_WINDOWS, Capability.ON_POLICY_ROLLOUTS})

    def validate_config(self) -> None:
        self.config.__post_init__()

    def training_step(self, batch: Mapping[str, Any]) -> Mapping[str, Tensor]:
        expert = self.discriminator(batch["expert_states"], batch["expert_actions"],
                                    batch["expert_task_ids"], batch["expert_valid_mask"])
        student = self.discriminator(batch["student_states"], batch["student_actions"],
                                     batch["student_task_ids"], batch["student_valid_mask"])
        loss = occupancy_discriminator_loss(expert, student, batch["expert_valid_mask"],
                                             batch["student_valid_mask"])
        self.step += 1
        return {"loss": loss, "expert_logits": expert, "student_logits": student}

    def sample_action_chunk(self, batch: Mapping[str, Any]) -> Tensor:
        raise RuntimeError("an occupancy discriminator is not an action policy")

    def save_method_state(self, path: str | Path) -> Path:
        return save_method_checkpoint(
            path, method_name=self.name, models={"occupancy_discriminator": self.discriminator},
            optimizers={"occupancy_discriminator": self.optimizer},
            schedulers={"occupancy_discriminator": self.scheduler}, scaler=None,
            counters={"step": self.step}, config=asdict(self.config), provenance={},
            trainable_names={"occupancy_discriminator": [n for n, p in self.discriminator.named_parameters() if p.requires_grad]},
        )

    def load_method_state(self, path: str | Path) -> Mapping[str, Any]:
        value = load_method_checkpoint(
            path, expected_method=self.name, models={"occupancy_discriminator": self.discriminator},
            optimizers={"occupancy_discriminator": self.optimizer},
            schedulers={"occupancy_discriminator": self.scheduler},
        )
        self.step = value["counters"]["step"]
        return value

    def dry_run_report(self) -> DryRunReport:
        if self._dry_run is None:
            raise RuntimeError("dry-run context was not provided")
        return self._dry_run


class OccupancyTMDMethod(ResearchMethod):
    name = "occupancy_tmd"
    classification = "proposed method"

    def __init__(
        self,
        *,
        tmd_stage2: ResearchMethod,
        discriminator: OccupancyWindowDiscriminator,
        config: OccupancyTMDConfig,
        gate: OccupancyGate,
        capabilities: CapabilitySet,
        dry_run: DryRunReport | None = None,
    ) -> None:
        capabilities.require(
            {Capability.PATH_WINDOWS, Capability.ON_POLICY_ROLLOUTS, Capability.FLOW_SCORE},
            method=self.name,
        )
        gate.require_open()
        self.tmd_stage2, self.discriminator, self.config, self.gate = (
            tmd_stage2,
            discriminator,
            config,
            gate,
        )
        self.step = 0
        self._dry_run = dry_run
        # Weighting consumes a separately trained, held-out-gated checkpoint.
        # It is frozen during downstream TMD updates.
        self.discriminator.requires_grad_(False)

    def required_data_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.PATH_WINDOWS, Capability.ON_POLICY_ROLLOUTS, Capability.FLOW_SCORE})

    def validate_config(self) -> None:
        self.config.__post_init__()
        self.gate.require_open()

    def training_step(self, batch: Mapping[str, Any]) -> Mapping[str, Tensor]:
        logits = self.discriminator(
            batch["states"], batch["actions"], batch["task_ids"], batch["valid_mask"]
        )
        weights = MismatchPrioritizationWeight.from_prefix_logits(
            logits,
            batch["valid_mask"],
            minimum=self.config.minimum_weight,
            maximum=self.config.maximum_weight,
        )
        downstream = dict(batch)
        downstream["prioritization_weight"] = weights.value
        result = dict(self.tmd_stage2.training_step(downstream))
        result["clipping_fraction"] = weights.clipping_fraction
        self.step += 1
        return result

    def sample_action_chunk(self, batch: Mapping[str, Any]) -> Tensor:
        return self.tmd_stage2.sample_action_chunk(batch)

    def save_method_state(self, path: str | Path) -> Path:
        target = Path(path)
        downstream_path = target.with_name(target.name + ".tmd_stage2.pt")
        self.tmd_stage2.save_method_state(downstream_path)
        downstream_hash = hashlib.sha256(downstream_path.read_bytes()).hexdigest()
        return save_method_checkpoint(
            target,
            method_name=self.name,
            models={"occupancy_discriminator": self.discriminator},
            optimizers={},
            schedulers={},
            scaler=None,
            counters={"step": self.step},
            config={
                **asdict(self.config),
                "gate": asdict(self.gate),
                "downstream_checkpoint": downstream_path.name,
                "downstream_checkpoint_sha256": downstream_hash,
            },
            provenance={},
            trainable_names={
                "occupancy_discriminator": [n for n, p in self.discriminator.named_parameters() if p.requires_grad]
            },
        )

    def load_method_state(self, path: str | Path) -> Mapping[str, Any]:
        value = load_method_checkpoint(
            path,
            expected_method=self.name,
            models={"occupancy_discriminator": self.discriminator},
            optimizers={},
            schedulers={},
        )
        self.step = value["counters"]["step"]
        downstream_path = Path(path).with_name(value["config"]["downstream_checkpoint"])
        if hashlib.sha256(downstream_path.read_bytes()).hexdigest() != value["config"][
            "downstream_checkpoint_sha256"
        ]:
            raise RuntimeError("occupancy-TMD downstream checkpoint hash mismatch")
        self.tmd_stage2.load_method_state(downstream_path)
        return value

    def dry_run_report(self) -> DryRunReport:
        if self._dry_run is None:
            raise RuntimeError("dry-run context was not provided")
        return self._dry_run
