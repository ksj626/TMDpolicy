"""Occupancy discriminator and occupancy-weighted Stage-1 TMD programs."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from tmd_policy.backends.action_coordinates import ActionNormalizer
from tmd_policy.methods.tmd.program import TMDStage1Program, sample_stage1_generator
from tmd_policy.training.engine import TrainingProgram

from .networks import OccupancyDiscriminator, WindowNormalizer


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
        occupancy_discriminator: OccupancyDiscriminator,
        occupancy_normalizer: WindowNormalizer,
        student_action_normalizer: ActionNormalizer,
        occupancy_config: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.occupancy_discriminator = occupancy_discriminator
        self.occupancy_normalizer = occupancy_normalizer
        self.student_action_normalizer = student_action_normalizer
        self.occupancy_config = dict(occupancy_config)
        self.occupancy_discriminator.requires_grad_(False).eval()
        self.occupancy_normalizer.requires_grad_(False)
        self.student_action_normalizer.requires_grad_(False)

    @staticmethod
    def _visual(batch: dict[str, Any], length: int, device: torch.device) -> Tensor:
        values = []
        for key in sorted(key for key in batch if key.startswith("observation.images.")):
            image = torch.as_tensor(batch[key], device=device, dtype=torch.float32)
            if image.ndim == 5:
                image = image[:, -1]
            values.append(image.mean(dim=(-2, -1)))
            if len(values) == 2:
                break
        if not values:
            raise ValueError("occupancy-weighted TMD requires real observation images")
        while len(values) < 2:
            values.append(torch.zeros_like(values[0]))
        return torch.cat(values, dim=-1)[:, None].expand(-1, length, -1)

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
        )
        canonical = self.student_action_normalizer.unnormalize(generated[..., :7])
        length = int(self.occupancy_config["window_length"])
        state = torch.as_tensor(batch["observation.state"], device=device, dtype=torch.float32)
        if state.ndim == 3:
            state = state[:, -1]
        state = state[:, None, :8].expand(-1, length, -1)
        action = canonical[:, :length]
        visual = self._visual(batch, length, device)
        path_valid = valid[:, :length]
        state, action, visual = self.occupancy_normalizer(state, action, visual)
        position = torch.arange(length, device=device)[None].expand(batch_size, -1)
        logits = self.occupancy_discriminator(
            state,
            action,
            visual,
            torch.as_tensor(batch["task_index"], device=device),
            position,
            path_valid,
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
            }
        )
        return value


__all__ = [
    "OccupancyDiscriminatorProgram",
    "OccupancyWeightedTMDProgram",
    "weighted_generator_loss",
]
