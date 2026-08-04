"""Paper-feature DMD2-flow with PI0.5 fake score, TTUR, VSD, and no SFT."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tmd_policy.backends.action_coordinates import ActionCoordinateBridge
from tmd_policy.backends.lerobot.pi05_teacher import LeRobotPI05Teacher, PI05ConditionCache
from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent, SmolVLAConditionCache
from tmd_policy.methods.discriminators import (
    CachedVLAFeatureDiscriminator,
    IntermediateFeatureDiscriminator,
)
from tmd_policy.methods.flow_objectives import (
    corrupt_rectified_flow,
    executable_coordinate_mask,
    sample_shifted_time,
    stopped_dmd2_direction,
    stopped_l1_score_direction,
    surrogate_vector_loss,
    validate_time_distribution,
)
from tmd_policy.training.engine import TrainingProgram

from .fake_scores import PI05CloneFakeScore, SmolVLACloneFakeScore
from .networks import ActionScoreTransformer


def _masked_mean(values: Tensor, valid_coordinates: Tensor) -> Tensor:
    mask = valid_coordinates.to(values.dtype)
    denominator = mask.flatten(1).sum(dim=1).clamp_min(1)
    return (values * mask).flatten(1).sum(dim=1) / denominator


@contextmanager
def _frozen_parameters(module: nn.Module):
    states = [parameter.requires_grad for parameter in module.parameters()]
    try:
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, state in zip(module.parameters(), states, strict=True):
            parameter.requires_grad_(state)


class DMD2FlowProgram(TrainingProgram):
    """Faithful objective: VSD plus noised-feature GAN, with no data regression."""

    def __init__(
        self,
        *,
        student: LeRobotSmolVLAStudent,
        teacher: LeRobotPI05Teacher,
        bridge: ActionCoordinateBridge,
        config: dict[str, Any],
        fake_score: nn.Module | None = None,
        preserve_student_trainability: bool = False,
    ) -> None:
        super().__init__()
        if "data_weight" in config:
            raise ValueError(
                "dmd2.data_weight was removed: faithful dmd2_flow contains only VSD and GAN; "
                "use an explicitly named SFT-hybrid ablation instead"
            )
        self.student = student
        self.teacher = teacher
        self.bridge = bridge
        self.dmd_config = dict(config)
        self.fake_variant = str(config["fake_score_variant"])
        if self.fake_variant == "lightweight":
            fake_cfg = config["fake_score"]
            self.fake_score: nn.Module = ActionScoreTransformer(
                num_tasks=int(fake_cfg["num_tasks"]),
                model_dim=int(fake_cfg["model_dim"]),
                layers=int(fake_cfg["layers"]),
                heads=int(fake_cfg["heads"]),
                feedforward_dim=int(fake_cfg["feedforward_dim"]),
            )
            self.fake_classification = "lightweight action-score ablation; not paper-faithful"
        elif self.fake_variant in {"pi05_clone", "smolvla_clone"}:
            expected = PI05CloneFakeScore if self.fake_variant == "pi05_clone" else SmolVLACloneFakeScore
            if not isinstance(fake_score, expected):
                raise TypeError(f"{self.fake_variant} requires {expected.__name__}")
            self.fake_score = fake_score
            self.fake_classification = fake_score.classification
        else:
            raise ValueError(f"unknown fake-score variant: {self.fake_variant}")

        discriminator = config["discriminator"]
        self.discriminator_variant = str(discriminator["variant"])
        self.selected_layers = tuple(int(index) for index in discriminator.get("selected_layers", ()))
        self.feature_source = str(discriminator.get("feature_source", "fake_score_features"))
        if self.discriminator_variant == "pi05_intermediate_features":
            if self.feature_source == "fake_score_features" and not isinstance(
                self.fake_score, PI05CloneFakeScore
            ):
                raise ValueError("fake_score_features requires the PI0.5 fake-score suffix")
            dimensions = {index: teacher.action_expert_feature_dim for index in self.selected_layers}
            self.discriminator: nn.Module = IntermediateFeatureDiscriminator(
                dimensions,
                hidden_dim=int(discriminator["hidden_dim"]),
                time_dim=int(discriminator.get("time_embedding_dim", 32)),
            )
            self._discriminator_device = (
                next(self.fake_score.parameters()).device
                if self.feature_source == "fake_score_features"
                else teacher.device
            )
        elif self.discriminator_variant == "cached_vla_features":
            condition_dim = 3 * int(student.flow.vlm_with_expert.config.text_config.hidden_size)
            self.discriminator = CachedVLAFeatureDiscriminator(
                condition_dim=condition_dim,
                num_tasks=int(discriminator["num_tasks"]),
                model_dim=int(discriminator["model_dim"]),
                layers=int(discriminator["layers"]),
                heads=int(discriminator["heads"]),
            )
            self._discriminator_device = next(student.parameters()).device
        else:
            raise ValueError(f"unknown discriminator variant: {self.discriminator_variant}")

        for section in ("vsd_time", "gan_time", "fake_score_time"):
            value = config[section]
            validate_time_distribution(
                float(value["minimum_time"]),
                float(value["maximum_time"]),
                float(value["time_shift_gamma"]),
            )
        if not preserve_student_trainability:
            self.student.configure_trainable(str(config["student_fine_tuning"]))
        self._fake_score_device = next(self.fake_score.parameters()).device

    def to(self, *args: Any, **kwargs: Any) -> "DMD2FlowProgram":
        super().to(*args, **kwargs)
        self.fake_score.to(self._fake_score_device)
        self.discriminator.to(self._discriminator_device)
        return self

    def phase_schedule(self) -> tuple[str, ...]:
        ratio = int(self.dmd_config["fake_updates_per_generator"])
        if ratio < 1:
            raise ValueError("fake_updates_per_generator must be positive")
        return ("fake",) * ratio + ("discriminator", "generator")

    @property
    def student_device(self) -> torch.device:
        return next(self.student.parameters()).device

    def _raw_state_task(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        state = torch.as_tensor(batch["observation.state"], device=self.student_device, dtype=torch.float32)
        task = torch.as_tensor(batch["task_index"], device=self.student_device).long().flatten()
        return state, task

    def _valid(self, batch: dict[str, Any], device: torch.device) -> Tensor:
        value = batch.get("action_is_pad")
        if value is None:
            batch_size = int(torch.as_tensor(batch["observation.state"]).shape[0])
            return torch.ones(batch_size, 50, device=device, dtype=torch.bool)
        return ~torch.as_tensor(value, device=device).bool()

    def _sample_time(self, name: str, batch_size: int, device: torch.device) -> Tensor:
        value = self.dmd_config[name]
        return sample_shifted_time(
            batch_size,
            device=device,
            dtype=torch.float32,
            minimum_time=float(value["minimum_time"]),
            maximum_time=float(value["maximum_time"]),
            gamma=float(value["time_shift_gamma"]),
        )

    def _sample_student(self, batch: dict[str, Any], *, requires_grad: bool) -> Tensor:
        processed = self.student.preprocess_observation(batch)
        condition = self.student.encode_condition(processed)
        noise = torch.randn(condition.batch_size, 50, 32, device=self.student_device, dtype=torch.float32)
        if requires_grad:
            return self.student.sample(condition, noise, int(self.dmd_config["generation_steps"]))
        with torch.no_grad():
            return self.student.sample(
                condition, noise, int(self.dmd_config["generation_steps"])
            ).detach()

    def _student_condition(self, batch: dict[str, Any]) -> SmolVLAConditionCache:
        return self.student.encode_condition(self.student.preprocess_observation(batch))

    def _teacher_condition(self, batch: dict[str, Any]) -> PI05ConditionCache:
        return self.teacher.encode_condition(self.teacher.preprocess_observation(batch))

    def _fake_condition(
        self, batch: dict[str, Any], teacher_condition: PI05ConditionCache | None = None
    ) -> Any:
        if self.fake_variant == "lightweight":
            return None
        if self.fake_variant == "pi05_clone" and teacher_condition is not None:
            return teacher_condition
        return self.fake_score.condition(batch)

    def _fake_velocity(
        self,
        batch: dict[str, Any],
        x_t: Tensor,
        time: Tensor,
        condition: Any | None = None,
    ) -> Tensor:
        if self.fake_variant == "lightweight":
            state, task = self._raw_state_task(batch)
            return self.fake_score(
                x_t.to(self.student_device), time.to(self.student_device), state, task
            ).to(x_t.device)
        target_device = next(self.fake_score.parameters()).device
        condition = condition if condition is not None else self._fake_condition(batch)
        return self.fake_score(condition, x_t.to(target_device), time.to(target_device)).to(x_t.device)

    @staticmethod
    def _denoised_prediction(x_t: Tensor, time: Tensor, velocity: Tensor) -> Tensor:
        return x_t - time[:, None, None] * velocity

    def _real_actions_teacher(self, batch: dict[str, Any], valid: Tensor) -> Tensor:
        processed = self.student.preprocess_observation(batch)
        student_actions = self.student.policy.prepare_action(processed).detach()
        return self.bridge.student_to_teacher(student_actions, valid).values

    def _fake_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        generated = self._sample_student(batch, requires_grad=False)
        valid_student = self._valid(batch, self.student_device)
        clean = self.bridge.student_to_teacher(generated, valid_student).values.detach()
        fake_device = self._fake_score_device
        clean = clean.to(fake_device)
        valid = executable_coordinate_mask(valid_student.to(fake_device), 32)
        time = self._sample_time("fake_score_time", clean.shape[0], fake_device)
        noise = torch.randn_like(clean)
        noised = corrupt_rectified_flow(clean, time, noise)
        target_velocity = noise - clean
        teacher_condition = self._teacher_condition(batch) if self.fake_variant == "pi05_clone" else None
        condition = self._fake_condition(batch, teacher_condition)
        prediction = self._fake_velocity(batch, noised, time, condition)
        per_sample = _masked_mean((prediction - target_velocity).square(), valid)
        loss = per_sample.mean()
        return loss, {"velocity_mse": float(loss.detach())}

    def _paper_features(
        self,
        condition: PI05ConditionCache,
        noised: Tensor,
        time: Tensor,
        *,
        require_input_grad: bool,
    ) -> dict[int, Tensor]:
        source: Any = self.fake_score if self.feature_source == "fake_score_features" else self.teacher
        if isinstance(source, PI05CloneFakeScore) and not require_input_grad:
            # DMD2 trains both the classifier heads and the fake-model feature
            # extractor during discriminator updates. The action is detached by
            # the caller, so this path creates parameter gradients only.
            return source._run(condition, noised, time, self.selected_layers)[1]
        context = _frozen_parameters(source) if isinstance(source, nn.Module) else torch.enable_grad()
        with context:
            return source.intermediate_features(
                condition,
                noised,
                time,
                self.selected_layers,
                require_input_grad,
            )

    def _paper_layer_logits(
        self,
        condition: PI05ConditionCache,
        noised: Tensor,
        time: Tensor,
        valid: Tensor,
        *,
        require_input_grad: bool,
    ) -> dict[int, Tensor]:
        features = self._paper_features(
            condition, noised, time, require_input_grad=require_input_grad
        )
        discriminator = self.discriminator
        if not isinstance(discriminator, IntermediateFeatureDiscriminator):
            raise TypeError("paper feature path requires IntermediateFeatureDiscriminator")
        return discriminator.layer_logits(features, time, valid)

    def _discriminator_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        generated = self._sample_student(batch, requires_grad=False)
        valid_student = self._valid(batch, self.student_device)
        real = self._real_actions_teacher(batch, valid_student)
        fake = self.bridge.student_to_teacher(generated, valid_student).values.detach()
        if self.discriminator_variant == "pi05_intermediate_features":
            device = self._discriminator_device
            real, fake = real.to(device), fake.to(device)
            time = self._sample_time("gan_time", real.shape[0], device)
            real_noised = corrupt_rectified_flow(real, time, torch.randn_like(real))
            fake_noised = corrupt_rectified_flow(fake, time, torch.randn_like(fake))
            condition = self._teacher_condition(batch)
            valid = valid_student.to(device)
            real_logits = self._paper_layer_logits(
                condition, real_noised, time, valid, require_input_grad=False
            )
            fake_logits = self._paper_layer_logits(
                condition, fake_noised, time, valid, require_input_grad=False
            )
            layer_losses = [
                F.softplus(-real_logits[index]).mean() + F.softplus(fake_logits[index]).mean()
                for index in self.selected_layers
            ]
            loss = torch.stack(layer_losses).mean()
            real_values = torch.stack(list(real_logits.values()))
            fake_values = torch.stack(list(fake_logits.values()))
        else:
            condition = self._student_condition(batch)
            device = self._discriminator_device
            time = self._sample_time("gan_time", real.shape[0], device)
            real_noised = corrupt_rectified_flow(real.to(device), time, torch.randn_like(real.to(device)))
            fake_noised = corrupt_rectified_flow(fake.to(device), time, torch.randn_like(fake.to(device)))
            task = torch.as_tensor(batch["task_index"], device=device).long().flatten()
            valid = valid_student.to(device)
            real_values = self.discriminator(real_noised, time, condition.condition_features, task, valid)
            fake_values = self.discriminator(fake_noised, time, condition.condition_features, task, valid)
            loss = F.softplus(-real_values).mean() + F.softplus(fake_values).mean()
        return loss, {
            "real_probability": float(real_values.sigmoid().mean().detach()),
            "fake_probability": float(fake_values.sigmoid().mean().detach()),
            "gan_noise_time": float(time.mean().detach()),
        }

    def _distribution_matching_loss(
        self, batch: dict[str, Any], generated: Tensor, valid_student: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        teacher_generated = self.bridge.student_to_teacher(generated, valid_student).values.to(
            self.teacher.device
        )
        valid_coordinates = executable_coordinate_mask(
            valid_student.to(self.teacher.device), teacher_generated.shape[-1]
        )
        time = self._sample_time("vsd_time", teacher_generated.shape[0], self.teacher.device)
        noised = corrupt_rectified_flow(teacher_generated, time, torch.randn_like(teacher_generated))
        condition = self._teacher_condition(batch)
        teacher_velocity = self.teacher.velocity(condition, noised.detach(), time).detach()
        with torch.no_grad():
            fake_velocity = self._fake_velocity(batch, noised.detach(), time, condition).detach()
        teacher_prediction = self._denoised_prediction(noised.detach(), time, teacher_velocity)
        fake_prediction = self._denoised_prediction(noised.detach(), time, fake_velocity)
        normalization = str(self.dmd_config["vsd_normalization"])
        if normalization == "dmd2_teacher_residual_mean_abs":
            direction, values = stopped_dmd2_direction(
                fake_prediction,
                teacher_prediction,
                teacher_generated.detach(),
                valid_coordinates,
                epsilon=float(self.dmd_config["vsd_normalization_epsilon"]),
            )
        elif normalization == "tmd_fake_teacher_difference_l1":
            direction, values = stopped_l1_score_direction(
                fake_prediction,
                teacher_prediction,
                valid_coordinates,
                epsilon=float(self.dmd_config["vsd_normalization_epsilon"]),
            )
        else:
            raise ValueError(f"unknown VSD normalization: {normalization}")
        loss = surrogate_vector_loss(teacher_generated, direction, valid_coordinates)
        return loss, {name: float(value.mean()) for name, value in values.items()}

    def _generator_gan_loss(
        self, batch: dict[str, Any], generated: Tensor, valid_student: Tensor
    ) -> Tensor:
        fake = self.bridge.student_to_teacher(generated, valid_student).values
        if self.discriminator_variant == "pi05_intermediate_features":
            device = self._discriminator_device
            fake = fake.to(device)
            time = self._sample_time("gan_time", fake.shape[0], device)
            noised = corrupt_rectified_flow(fake, time, torch.randn_like(fake))
            condition = self._teacher_condition(batch)
            with _frozen_parameters(self.discriminator):
                logits = self._paper_layer_logits(
                    condition,
                    noised,
                    time,
                    valid_student.to(device),
                    require_input_grad=True,
                )
                loss = torch.stack([F.softplus(-value).mean() for value in logits.values()]).mean()
        else:
            condition = self._student_condition(batch)
            device = self._discriminator_device
            fake = fake.to(device)
            time = self._sample_time("gan_time", fake.shape[0], device)
            noised = corrupt_rectified_flow(fake, time, torch.randn_like(fake))
            task = torch.as_tensor(batch["task_index"], device=device).long().flatten()
            with _frozen_parameters(self.discriminator):
                logits = self.discriminator(
                    noised,
                    time,
                    condition.condition_features,
                    task,
                    valid_student.to(device),
                )
                loss = F.softplus(-logits).mean()
        action_gradient = torch.autograd.grad(loss, generated, retain_graph=True, allow_unused=True)[0]
        if action_gradient is None or not torch.isfinite(action_gradient).all() or action_gradient.abs().sum() == 0:
            raise RuntimeError("generator GAN path produced no nonzero finite gradient to fake actions")
        if any(parameter.grad is not None for parameter in self.teacher.policy.parameters()):
            raise RuntimeError("frozen PI0.5 teacher unexpectedly received gradients")
        return loss

    def _generator_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        generated = self._sample_student(batch, requires_grad=True)
        valid = self._valid(batch, self.student_device)
        vsd, metrics = self._distribution_matching_loss(batch, generated, valid)
        gan = self._generator_gan_loss(batch, generated, valid)
        total = vsd + float(self.dmd_config["gan_weight"]) * gan
        return total, {
            **metrics,
            "distribution_matching": float(vsd.detach()),
            "gan": float(gan.detach()),
        }

    def loss(self, batch: dict[str, Any], phase: str) -> tuple[Tensor, dict[str, float]]:
        if phase == "fake":
            return self._fake_loss(batch)
        if phase == "discriminator":
            return self._discriminator_loss(batch)
        if phase == "generator":
            return self._generator_loss(batch)
        raise ValueError(f"unknown DMD2 phase: {phase}")

    def validation_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        return self._fake_loss(batch)

    def make_optimizers(self, training: dict[str, Any]) -> dict[str, torch.optim.Optimizer]:
        common = {
            "betas": tuple(training.get("betas", [0.9, 0.95])),
            "weight_decay": float(training["weight_decay"]),
        }
        fake_parameters = [parameter for parameter in self.fake_score.parameters() if parameter.requires_grad]
        discriminator_parameters = [parameter for parameter in self.discriminator.parameters() if parameter.requires_grad]
        if self.feature_source == "fake_score_features":
            discriminator_parameters = [*discriminator_parameters, *fake_parameters]
        return {
            "fake": torch.optim.AdamW(
                fake_parameters, lr=float(self.dmd_config["fake_score_learning_rate"]), **common
            ),
            "discriminator": torch.optim.AdamW(
                discriminator_parameters,
                lr=float(self.dmd_config["discriminator_learning_rate"]),
                **common,
            ),
            "generator": torch.optim.AdamW(
                [parameter for parameter in self.student.parameters() if parameter.requires_grad],
                lr=float(self.dmd_config["generator_learning_rate"]),
                **common,
            ),
        }

    def extra_provenance(self) -> dict[str, Any]:
        if isinstance(self.discriminator, IntermediateFeatureDiscriminator):
            discriminator = self.discriminator.provenance(feature_source=self.feature_source)
        else:
            discriminator = self.discriminator.provenance(
                self.student.condition_feature_identity
            )
        return {
            "dmd2": self.dmd_config,
            "objective": "L_VSD + gan_weight * L_GAN; no SFT/data loss",
            "fake_score_classification": self.fake_classification,
            "fake_score_parameters": getattr(self.fake_score, "parameter_report", None),
            "discriminator": discriminator,
            "teacher_device": str(self.teacher.device),
            "sampler_identity": "LeRobotSmolVLAStudent.sample differentiable Euler simulation",
            "resource_model": self.dmd_config.get("resource_estimate", {}),
        }


__all__ = ["DMD2FlowProgram"]
