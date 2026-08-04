from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tmd_policy.common.capabilities import Capability
from tmd_policy.common.checkpointing import load_method_checkpoint, save_method_checkpoint
from tmd_policy.methods.base import DryRunReport, ResearchMethod


@dataclass(frozen=True)
class FlowSFTConfig:
    fine_tuning: str = "frozen_backbone"
    mixed_precision: str = "bf16"
    gradient_accumulation: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4

    def __post_init__(self) -> None:
        if self.fine_tuning not in {"frozen_backbone", "lora", "full"}:
            raise ValueError("fine_tuning must be frozen_backbone, lora, or full")
        if self.mixed_precision not in {"none", "bf16", "fp16"}:
            raise ValueError("unsupported mixed precision")
        if self.gradient_accumulation < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer/accumulation configuration")


def flow_sft_loss(
    velocity_fn: Callable[[Tensor, Tensor], Tensor],
    actions: Tensor,
    noise: Tensor,
    time: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    if actions.shape != noise.shape or valid_mask.shape != actions.shape[:2]:
        raise ValueError("actions/noise [B,H,D] and valid_mask [B,H] shapes must agree")
    if valid_mask.dtype != torch.bool or torch.any(valid_mask.sum(dim=1) == 0):
        raise ValueError("every Flow-SFT sample needs at least one valid action")
    expanded_time = time
    while expanded_time.ndim < actions.ndim:
        expanded_time = expanded_time.unsqueeze(-1)
    action_t = (1 - expanded_time) * actions + expanded_time * noise
    target = noise - actions
    predicted = velocity_fn(action_t, time)
    losses = (predicted - target).square() * valid_mask.unsqueeze(-1)
    denominator = valid_mask.sum(dim=1) * actions.shape[-1]
    return losses.sum(dim=(1, 2)) / denominator


def configure_trainable_parameters(policy: nn.Module, mode: str) -> list[str]:
    for parameter in policy.parameters():
        parameter.requires_grad_(mode == "full")
    if mode == "frozen_backbone":
        markers = ("action", "expert", "state_proj", "time_mlp")
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_(any(marker in name.lower() for marker in markers))
    elif mode == "lora":
        for name, parameter in policy.named_parameters():
            parameter.requires_grad_("lora" in name.lower())
        if not any(parameter.requires_grad for parameter in policy.parameters()):
            raise RuntimeError("LoRA mode requested but the policy has no LoRA parameters")
    elif mode != "full":
        raise ValueError(f"unknown fine-tuning mode: {mode}")
    names = sorted(name for name, parameter in policy.named_parameters() if parameter.requires_grad)
    if not names:
        raise RuntimeError("Flow-SFT selected no trainable parameters")
    return names


class FlowSFTMethod(ResearchMethod):
    name = "flow_sft"
    classification = "exact pinned SmolVLA objective"

    def __init__(
        self,
        *,
        policy: nn.Module,
        config: FlowSFTConfig | None = None,
        provenance: Mapping[str, Any] | None = None,
        dry_run: DryRunReport | None = None,
    ) -> None:
        self.policy = policy
        self.config = config or FlowSFTConfig()
        self.trainable_names = configure_trainable_parameters(policy, self.config.fine_tuning)
        self.optimizer = torch.optim.AdamW(
            (parameter for parameter in policy.parameters() if parameter.requires_grad),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.config.mixed_precision == "fp16" and torch.cuda.is_available()
        )
        self.step = 0
        self.optimizer_steps = 0
        self.data_epoch = 0
        self.batch_in_epoch = 0
        self._dry_run = dry_run
        self.provenance = dict(provenance or {})

    def required_data_capabilities(self) -> frozenset[Capability]:
        return frozenset({Capability.EXPERT_ACTION_CHUNKS, Capability.FLOW_VELOCITY})

    def validate_config(self) -> None:
        self.config.__post_init__()

    def training_step(self, batch: Mapping[str, Any]) -> Mapping[str, Tensor]:
        if "action_is_pad" in batch and torch.any((~batch["action_is_pad"].bool()).sum(1) == 0):
            raise ValueError("every Flow-SFT sample needs at least one valid action")
        loss, _ = self.policy(dict(batch), reduction="none")
        if loss.ndim != 1:
            raise RuntimeError("pinned SmolVLA must return per-sample loss for Flow-SFT")
        self.step += 1
        return {"loss": loss.mean(), "per_sample_loss": loss}

    def sample_action_chunk(self, batch: Mapping[str, Any]) -> Tensor:
        return self.policy.predict_action_chunk(dict(batch))

    def save_method_state(self, path: str | Path) -> Path:
        return save_method_checkpoint(
            path,
            method_name=self.name,
            models={"student": self.policy},
            optimizers={"student": self.optimizer},
            schedulers={"student": self.scheduler},
            scaler=self.scaler,
            counters={
                "step": self.step,
                "optimizer_steps": self.optimizer_steps,
                "data_epoch": self.data_epoch,
                "batch_in_epoch": self.batch_in_epoch,
            },
            config=asdict(self.config),
            provenance=self.provenance,
            trainable_names={"student": self.trainable_names},
        )

    def load_method_state(self, path: str | Path) -> Mapping[str, Any]:
        value = load_method_checkpoint(
            path,
            expected_method=self.name,
            models={"student": self.policy},
            optimizers={"student": self.optimizer},
            schedulers={"student": self.scheduler},
            scaler=self.scaler,
        )
        self.step = value["counters"]["step"]
        self.optimizer_steps = value["counters"]["optimizer_steps"]
        self.data_epoch = value["counters"]["data_epoch"]
        self.batch_in_epoch = value["counters"]["batch_in_epoch"]
        if value["trainable_names"]["student"] != self.trainable_names:
            raise RuntimeError("trainable parameter names changed across resume")
        return value

    def dry_run_report(self) -> DryRunReport:
        if self._dry_run is None:
            raise RuntimeError("dry-run context was not provided")
        return self._dry_run
