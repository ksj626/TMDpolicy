"""Occupancy discriminator and occupancy-weighted Stage-1 TMD programs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from tmd_policy.backends.action_coordinates import ActionCoordinateBridge
from tmd_policy.backends.lerobot.pi05_teacher import LeRobotPI05Teacher
from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent
from tmd_policy.methods.discriminators import CachedVLAFeatureDiscriminator, IntermediateFeatureDiscriminator
from tmd_policy.methods.flow_objectives import (
    corrupt_rectified_flow,
    sample_shifted_time,
    validate_time_distribution,
)
from tmd_policy.methods.tmd.program import TMDStage1Program, sample_stage1_generator
from tmd_policy.training.engine import TrainingProgram

from .networks import OccupancyDiscriminator, WindowNormalizer


class ReplanOccupancyDiscriminatorProgram(TrainingProgram):
    """Joint short-window ratio over actual `(s,o,planned action,task)` replans."""

    def __init__(
        self,
        *,
        teacher: LeRobotPI05Teacher | None,
        student: LeRobotSmolVLAStudent,
        bridge: ActionCoordinateBridge | None,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self.teacher = teacher
        # The frozen feature encoder is an immutable external asset, not part of
        # the discriminator checkpoint or optimizer state.
        object.__setattr__(self, "student", student)
        self.bridge = bridge
        self.config = dict(config)
        self.variant = str(config["variant"])
        validate_time_distribution(
            float(config["minimum_time"]),
            float(config["maximum_time"]),
            float(config["time_shift_gamma"]),
        )
        if self.variant == "pi05_intermediate_features":
            if teacher is None or bridge is None:
                raise ValueError("paper occupancy discriminator requires PI0.5 teacher and coordinate bridge")
            self.selected_layers = tuple(int(value) for value in config["selected_layers"])
            self.discriminator: torch.nn.Module = IntermediateFeatureDiscriminator(
                {index: teacher.action_expert_feature_dim for index in self.selected_layers},
                hidden_dim=int(config["hidden_dim"]),
                time_dim=int(config.get("time_embedding_dim", 32)),
            )
            self._discriminator_device = teacher.device
        elif self.variant == "cached_vla_features":
            self.selected_layers = ()
            condition_dim = 3 * int(student.flow.vlm_with_expert.config.text_config.hidden_size)
            self.discriminator = CachedVLAFeatureDiscriminator(
                condition_dim=condition_dim,
                num_tasks=int(config["num_tasks"]),
                model_dim=int(config["model_dim"]),
                layers=int(config["layers"]),
                heads=int(config["heads"]),
            )
            self._discriminator_device = student.device
        else:
            raise ValueError(f"unknown occupancy discriminator variant: {self.variant}")
        self.student.requires_grad_(False).eval()

    def to(self, *args: Any, **kwargs: Any) -> "ReplanOccupancyDiscriminatorProgram":
        super().to(*args, **kwargs)
        self.discriminator.to(self._discriminator_device)
        return self

    def phase_schedule(self) -> tuple[str, ...]:
        return ("discriminator",)

    def loss(self, batch: dict[str, Any], phase: str) -> tuple[Tensor, dict[str, float]]:
        if phase != "discriminator":
            raise ValueError(f"occupancy discriminator has no phase {phase!r}")
        device = self._discriminator_device
        clean = torch.as_tensor(batch["action"], device=self.student.device, dtype=torch.float32)
        valid = torch.as_tensor(batch["action_valid"], device=self.student.device).bool()
        if self.variant == "pi05_intermediate_features":
            assert self.bridge is not None
            internal_actions = self.bridge.canonical_to_teacher(clean, valid).values.to(device)
        else:
            processed = self.student.preprocess_observation(batch)
            internal_actions = self.student.policy.prepare_action(processed).detach().to(device)
        time = sample_shifted_time(
            clean.shape[0],
            device=device,
            dtype=torch.float32,
            minimum_time=float(self.config["minimum_time"]),
            maximum_time=float(self.config["maximum_time"]),
            gamma=float(self.config["time_shift_gamma"]),
        )
        noised = corrupt_rectified_flow(internal_actions, time, torch.randn_like(internal_actions))
        valid_device = valid.to(device)
        labels = torch.as_tensor(batch["source_label"], device=device).float().flatten()
        if self.variant == "pi05_intermediate_features":
            assert self.teacher is not None
            condition = self.teacher.encode_condition(self.teacher.preprocess_observation(batch))
            features = self.teacher.intermediate_features(
                condition, noised, time, self.selected_layers, require_input_grad=False
            )
            layer_logits = self.discriminator.layer_logits(features, time, valid_device)
            loss = torch.stack(
                [
                    torch.where(labels.bool(), F.softplus(-value), F.softplus(value)).mean()
                    for value in layer_logits.values()
                ]
            ).mean()
            logits = torch.stack(list(layer_logits.values())).mean(dim=0)
        else:
            condition = self.student.encode_condition(self.student.preprocess_observation(batch))
            tasks = torch.as_tensor(batch["task_index"], device=device).long().flatten()
            logits = self.discriminator(noised, time, condition.condition_features, tasks, valid_device)
            loss = torch.where(labels.bool(), F.softplus(-logits), F.softplus(logits)).mean()
        predictions = logits >= 0
        metrics = {
            "accuracy": float((predictions == labels.bool()).float().mean().detach()),
            "noise_time": float(time.mean().detach()),
            "expert_probability": float(logits[labels.bool()].sigmoid().mean().detach())
            if labels.bool().any()
            else float("nan"),
            "student_probability": float(logits[~labels.bool()].sigmoid().mean().detach())
            if (~labels.bool()).any()
            else float("nan"),
        }
        tasks = torch.as_tensor(batch["task_index"], device=device).long().flatten()
        for task in tasks.unique().tolist():
            rows = tasks == task
            metrics[f"task_{task:02d}_accuracy"] = float(
                (predictions[rows] == labels[rows].bool()).float().mean().detach()
            )
        return loss, metrics

    def make_optimizers(self, training: dict[str, Any]) -> dict[str, torch.optim.Optimizer]:
        return {
            "discriminator": torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=float(training["learning_rate"]),
                betas=tuple(training.get("betas", [0.9, 0.95])),
                weight_decay=float(training["weight_decay"]),
            )
        }

    def extra_provenance(self) -> dict[str, Any]:
        provenance = {
            "occupancy_semantics": "joint short-window expert/student ratio at actual replans",
            "ratio": "log rho_E(s,o,a_plan,task) / rho_S(s,o,a_plan,task) under balanced priors",
            "variant": self.variant,
            "selected_layers": list(self.selected_layers),
            "rollout_classification": "historical fixed-checkpoint behavior occupancy, not exact current on-policy",
            "action_corruption": "a_tau=(1-tau)*a+tau*epsilon",
        }
        if self.variant == "cached_vla_features":
            provenance["discriminator"] = self.discriminator.provenance(
                self.student.condition_feature_identity
            )
        else:
            provenance["discriminator"] = {
                "variant": self.variant,
                "feature_source": "teacher_features",
                "selected_layers": list(self.selected_layers),
            }
        return provenance


def weighted_generator_loss(per_sample_loss: Tensor, weights: Tensor) -> Tensor:
    """Normalized detached importance weights; gradients still scale per sample."""

    if per_sample_loss.shape != weights.shape or torch.any(weights <= 0):
        raise ValueError("positive occupancy weights must match per-sample generator losses")
    normalized = weights.detach() / weights.detach().mean().clamp_min(1e-8)
    return (per_sample_loss * normalized).mean()


class OccupancyDiscriminatorProgram(TrainingProgram):
    def __init__(
        self,
        discriminator: OccupancyDiscriminator,
        normalizer: WindowNormalizer,
    ) -> None:
        super().__init__()
        self.discriminator = discriminator
        self.normalizer = normalizer

    def phase_schedule(self) -> tuple[str, ...]:
        return ("discriminator",)

    def _forward(self, batch: dict[str, Any]) -> Tensor:
        device = next(self.discriminator.parameters()).device
        state = torch.as_tensor(batch["state"], device=device, dtype=torch.float32)
        action = torch.as_tensor(batch["action"], device=device, dtype=torch.float32)
        visual = torch.as_tensor(batch["visual"], device=device, dtype=torch.float32)
        state, action, visual = self.normalizer(state, action, visual)
        return self.discriminator(
            state,
            action,
            visual,
            torch.as_tensor(batch["task_index"], device=device),
            torch.as_tensor(batch["position"], device=device),
            torch.as_tensor(batch["valid"], device=device).bool(),
        )

    def loss(self, batch: dict[str, Any], phase: str) -> tuple[Tensor, dict[str, float]]:
        if phase != "discriminator":
            raise ValueError(f"occupancy discriminator has no phase {phase!r}")
        logits = self._forward(batch)
        device = logits.device
        labels = torch.as_tensor(batch["source_label"], device=device, dtype=logits.dtype).flatten()
        balance = torch.as_tensor(batch["balance_weight"], device=device, dtype=logits.dtype).flatten()
        per_sample = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = (per_sample * balance).sum() / balance.sum().clamp_min(1e-8)
        predictions = logits >= 0
        return loss, {
            "accuracy": float((predictions == labels.bool()).float().mean().detach()),
            "expert_probability": float(logits[labels.bool()].sigmoid().mean().detach())
            if labels.bool().any()
            else float("nan"),
            "student_probability": float(logits[~labels.bool()].sigmoid().mean().detach())
            if (~labels.bool()).any()
            else float("nan"),
        }

    def make_optimizers(self, training: dict[str, Any]) -> dict[str, torch.optim.Optimizer]:
        return {
            "discriminator": torch.optim.AdamW(
                self.discriminator.parameters(),
                lr=float(training["learning_rate"]),
                betas=tuple(training.get("betas", [0.9, 0.95])),
                weight_decay=float(training["weight_decay"]),
            )
        }

    def extra_provenance(self) -> dict[str, Any]:
        return {
            "occupancy_orientation": "expert=1, student=0",
            "normalization_source": "train expert and train student rollout windows only",
            "normalization_samples": self.normalizer.fitted_samples,
            "balancing": "inverse joint frequency of task, within-window start position, and source",
        }


class OccupancyWeightedTMDProgram(TMDStage1Program):
    def __init__(
        self,
        *args: Any,
        occupancy_discriminator: torch.nn.Module,
        occupancy_teacher: LeRobotPI05Teacher | None,
        occupancy_bridge: ActionCoordinateBridge | None,
        occupancy_variant: str,
        occupancy_config: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.occupancy_discriminator = occupancy_discriminator
        self.occupancy_teacher = occupancy_teacher
        self.occupancy_bridge = occupancy_bridge
        self.occupancy_variant = occupancy_variant
        self.occupancy_config = dict(occupancy_config)
        self.occupancy_discriminator.requires_grad_(False).eval()
        if self.occupancy_variant == "pi05_intermediate_features":
            if occupancy_teacher is None or occupancy_bridge is None:
                raise ValueError("paper occupancy weighting requires its pinned PI0.5 feature backend")
        elif self.occupancy_variant != "cached_vla_features":
            raise ValueError(f"unknown occupancy weighting variant: {self.occupancy_variant}")

    @torch.no_grad()
    def occupancy_weights(self, batch: dict[str, Any], condition: Any, valid: Tensor) -> Tensor:
        device = next(self.head.parameters()).device
        batch_size = valid.shape[0]
        noise = torch.randn(batch_size, 50, 32, device=device)
        generated = sample_stage1_generator(
            self.student,
            self.head,
            condition,
            noise,
            outer_steps=int(self.occupancy_config["weight_sampler_outer_steps"]),
            inner_steps=int(self.occupancy_config["weight_sampler_inner_steps"]),
            student_time_shift_gamma=float(self.tmd_config["student_time_shift_gamma"]),
        )
        plan_valid = torch.ones(batch_size, 50, device=device, dtype=torch.bool)
        time = sample_shifted_time(
            batch_size,
            device=device,
            dtype=torch.float32,
            minimum_time=float(self.occupancy_config["minimum_time"]),
            maximum_time=float(self.occupancy_config["maximum_time"]),
            gamma=float(self.occupancy_config["time_shift_gamma"]),
        )
        if self.occupancy_variant == "pi05_intermediate_features":
            assert self.occupancy_teacher is not None and self.occupancy_bridge is not None
            teacher_device = self.occupancy_teacher.device
            teacher_action = self.occupancy_bridge.student_to_teacher(generated, plan_valid).values.to(
                teacher_device
            )
            teacher_time = time.to(teacher_device)
            noised = corrupt_rectified_flow(
                teacher_action, teacher_time, torch.randn_like(teacher_action)
            )
            teacher_condition = self.occupancy_teacher.encode_condition(
                self.occupancy_teacher.preprocess_observation(batch)
            )
            layers = tuple(int(value) for value in self.occupancy_config["selected_layers"])
            features = self.occupancy_teacher.intermediate_features(
                teacher_condition, noised, teacher_time, layers, require_input_grad=False
            )
            logits = torch.stack(
                list(
                    self.occupancy_discriminator.layer_logits(
                        features, teacher_time, plan_valid.to(teacher_device)
                    ).values()
                )
            ).mean(dim=0)
        else:
            noised = corrupt_rectified_flow(generated, time, torch.randn_like(generated))
            logits = self.occupancy_discriminator(
                noised,
                time,
                condition.condition_features,
                torch.as_tensor(batch["task_index"], device=device).long().flatten(),
                plan_valid,
            )
        weights = torch.exp(logits / float(self.occupancy_config["temperature"]))
        return weights.clamp(
            float(self.occupancy_config["minimum_weight"]),
            float(self.occupancy_config["maximum_weight"]),
        )

    def loss(self, batch: dict[str, Any], phase: str) -> tuple[Tensor, dict[str, float]]:
        if phase != "generator":
            raise ValueError(f"occupancy TMD has no phase {phase!r}")
        values, samples, valid, condition = self.compute_meanflow(batch)
        weights = self.occupancy_weights(batch, condition, valid)
        loss = weighted_generator_loss(values["per_sample_loss"], weights)
        return loss, {
            "weighted_meanflow": float(loss.detach()),
            "unweighted_meanflow": float(values["loss"].detach()),
            "weight_mean": float(weights.mean()),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "r_equals_s_fraction": float(samples.flow_matching_rows.float().mean()),
        }

    def extra_provenance(self) -> dict[str, Any]:
        value = super().extra_provenance()
        value.update(
            {
                "occupancy_weighting": self.occupancy_config,
                "weight_definition": "clipped exp(expert-vs-student discriminator logit / temperature)",
                "weight_gradient": "weights detached; per-sample generator gradient scaled before reduction",
                "occupancy_variant": self.occupancy_variant,
                "occupancy_unit": "actual current observation plus full generated [50,7] plan",
            }
        )
        return value


__all__ = [
    "OccupancyDiscriminatorProgram",
    "OccupancyWeightedTMDProgram",
    "ReplanOccupancyDiscriminatorProgram",
    "weighted_generator_loss",
]
