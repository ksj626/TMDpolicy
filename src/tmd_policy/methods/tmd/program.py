"""Real Stage-1 TMD program over SmolVLA action transitions."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent, SmolVLAConditionCache
from tmd_policy.training.engine import TrainingProgram

from .heads import GRUMeanFlowHead, SplitTransformerMeanFlowHead
from .meanflow import integrate_inner_flow, meanflow_loss, sample_meanflow_batch


def sample_stage1_generator(
    student: LeRobotSmolVLAStudent,
    head: nn.Module,
    condition: SmolVLAConditionCache,
    outer_noise: Tensor,
    *,
    outer_steps: int,
    inner_steps: int,
    inner_noises: Tensor | None = None,
) -> Tensor:
    """The single Stage-1 sampler used by Stage 1, Stage 2, and evaluation."""

    if outer_steps < 1 or inner_steps < 1:
        raise ValueError("outer_steps and inner_steps must be positive")
    if inner_noises is None:
        inner_noises = torch.randn(
            outer_steps, *outer_noise.shape, device=outer_noise.device, dtype=outer_noise.dtype
        )
    if inner_noises.shape != (outer_steps, *outer_noise.shape):
        raise ValueError("inner_noises must be [outer_steps,B,50,32]")
    grid = torch.linspace(1.0, 0.0, outer_steps + 1, device=outer_noise.device, dtype=outer_noise.dtype)
    value = outer_noise
    for index, (current, target) in enumerate(zip(grid[:-1], grid[1:], strict=True)):
        time = current.expand(value.shape[0])
        base_velocity, features = student.velocity_with_features(condition, value, time)
        transition = integrate_inner_flow(
            head,
            inner_noises[index],
            features,
            num_steps=inner_steps,
            base_velocity=base_velocity,
        )
        value = value + (target - current) * transition
    return value


class TMDStage1Program(TrainingProgram):
    def __init__(self, student: LeRobotSmolVLAStudent, config: dict[str, Any]) -> None:
        super().__init__()
        self.student = student
        self.tmd_config = dict(config)
        feature_dim = int(student.flow.vlm_with_expert.expert_hidden_size)
        variant = config["variant"]
        if variant == "tmd_split_transformer_head":
            self.head: nn.Module = SplitTransformerMeanFlowHead(
                action_dim=32,
                context_dim=feature_dim,
                model_dim=int(config["model_dim"]),
                layers=int(config["layers"]),
                heads=int(config["heads"]),
                feedforward_dim=int(config["feedforward_dim"]),
                dropout=float(config.get("dropout", 0.0)),
                attention_mode=str(config.get("attention_mode", "bidirectional")),
            )
        elif variant == "tmd_gru_head":
            self.head = GRUMeanFlowHead(
                action_dim=32,
                context_dim=feature_dim,
                hidden_dim=int(config["model_dim"]),
                layers=int(config["layers"]),
            )
        else:
            raise ValueError(f"unknown TMD Stage-1 variant: {variant}")

        expert_layers = list(student.flow.vlm_with_expert.lm_expert.layers)
        last_k = int(config["last_k_expert_blocks"])
        if not 1 <= last_k < len(expert_layers):
            raise ValueError(f"last_k_expert_blocks must be in [1,{len(expert_layers)-1}]")
        self.early_block_names = tuple(
            f"policy.model.vlm_with_expert.lm_expert.layers.{index}"
            for index in range(len(expert_layers) - last_k)
        )
        self.flow_block_names = tuple(
            f"policy.model.vlm_with_expert.lm_expert.layers.{index}"
            for index in range(len(expert_layers) - last_k, len(expert_layers))
        )
        student.configure_tmd_trainable(
            last_k_expert_blocks=last_k,
            train_backbone_flow_blocks=bool(config.get("train_backbone_flow_blocks", True)),
        )
        self.head.requires_grad_(True)
        self.intended_student_trainable_names = tuple(student.trainable_parameter_names)

    def phase_schedule(self) -> tuple[str, ...]:
        return ("generator",)

    @staticmethod
    def _coordinate_mask(valid: Tensor, width: int = 32) -> Tensor:
        dimensions = torch.arange(width, device=valid.device) < 7
        return valid.unsqueeze(-1) & dimensions[None, None]

    def _batch_components(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, Any], SmolVLAConditionCache, Tensor, Tensor]:
        processed = self.student.preprocess_observation(batch)
        condition = self.student.encode_condition(processed)
        actions = self.student.policy.prepare_action(processed)
        valid = ~processed.get(
            "action_is_pad", torch.zeros(actions.shape[:2], device=actions.device, dtype=torch.bool)
        ).bool()
        return processed, condition, actions, valid

    def compute_meanflow(
        self, batch: dict[str, Any]
    ) -> tuple[dict[str, Tensor], Any, Tensor, SmolVLAConditionCache]:
        _, condition, actions, valid = self._batch_components(batch)
        samples = sample_meanflow_batch(
            actions,
            flow_matching_fraction=float(self.tmd_config["flow_matching_fraction"]),
            outer_time_shift_gamma=float(self.tmd_config.get("outer_time_shift_gamma", 1.0)),
            inner_time_shift_gamma=float(self.tmd_config.get("inner_time_shift_gamma", 1.0)),
            discrete_target_steps=int(self.tmd_config.get("inference_inner_steps", 1)),
        )
        outer_time = samples.outer_time
        x_t = (1.0 - outer_time[:, None, None]) * actions + outer_time[:, None, None] * samples.outer_noise
        base_velocity, features = self.student.velocity_with_features(condition, x_t, outer_time)
        dropout = float(self.tmd_config.get("condition_dropout_probability", 0.0))
        if not 0.0 <= dropout < 1.0:
            raise ValueError("condition_dropout_probability must be in [0,1)")
        if dropout:
            keep = (torch.rand(actions.shape[0], device=actions.device) >= dropout).to(features.dtype)
            features = features * keep[:, None, None]
        true_transition = samples.outer_noise - actions
        values = meanflow_loss(
            self.head,
            target_transition=true_transition,
            inner_source=samples.inner_source,
            inner_time=samples.inner_time,
            target_time=samples.target_time,
            context=features,
            base_velocity=base_velocity,
            valid_coordinates=self._coordinate_mask(valid),
            normalization_constant=float(self.tmd_config["normalization_constant"]),
        )
        return values, samples, valid, condition

    def loss(self, batch: dict[str, Any], phase: str) -> tuple[Tensor, dict[str, float]]:
        if phase != "generator":
            raise ValueError(f"TMD Stage 1 has no phase {phase!r}")
        values, samples, _, _ = self.compute_meanflow(batch)
        return values["loss"], {
            "meanflow": float(values["loss"].detach()),
            "r_equals_s_fraction": float(samples.flow_matching_rows.float().mean()),
            "inner_outer_source_cosine": float(
                torch.nn.functional.cosine_similarity(
                    samples.outer_noise.flatten(1), samples.inner_source.flatten(1)
                ).mean()
            ),
            "base_velocity_abs": float(values["base_velocity"].abs().mean().detach()),
            "residual_abs": float(values["residual"].abs().mean().detach()),
        }

    def sample(
        self,
        condition: SmolVLAConditionCache,
        outer_noise: Tensor,
        *,
        outer_steps: int,
        inner_steps: int,
        inner_noises: Tensor | None = None,
    ) -> Tensor:
        """Stage-1 generator shared by training simulation, Stage 2, and evaluation."""

        return sample_stage1_generator(
            self.student,
            self.head,
            condition,
            outer_noise,
            outer_steps=outer_steps,
            inner_steps=inner_steps,
            inner_noises=inner_noises,
        )

    def make_optimizers(self, training: dict[str, Any]) -> dict[str, torch.optim.Optimizer]:
        return {
            "generator": torch.optim.AdamW(
                [parameter for parameter in self.parameters() if parameter.requires_grad],
                lr=float(training["learning_rate"]),
                betas=tuple(training.get("betas", [0.9, 0.95])),
                eps=float(training.get("epsilon", 1e-8)),
                weight_decay=float(training["weight_decay"]),
            )
        }

    def extra_provenance(self) -> dict[str, Any]:
        return {
            "tmd_stage1": self.tmd_config,
            "architecture_classification": (
                "paper-closest SmolVLA split-transformer action-space port; not an exact paper architecture"
                if self.tmd_config["variant"] == "tmd_split_transformer_head"
                else "lightweight GRU architectural adaptation; not paper-faithful"
            ),
            "frozen_early_expert_blocks": list(self.early_block_names),
            "outer_flow_blocks": list(self.flow_block_names),
            "inner_update_modules": [name for name, _ in self.head.named_parameters()],
            "intended_trainable_parameter_names": list(self.intended_student_trainable_names),
            "attention_mode": getattr(self.head, "attention_mode", "recurrent"),
            "preconditioning": "u=z-(v_smolvla+Delta); zero Delta reproduces SmolVLA transition",
        }


__all__ = ["TMDStage1Program", "sample_stage1_generator"]
