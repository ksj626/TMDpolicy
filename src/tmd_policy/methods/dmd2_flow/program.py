"""Paper-feature DMD2-flow with PI0.5 fake score, TTUR, VSD, and no SFT."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tmd_policy.backends.action_coordinates import ActionCoordinateBridge, safe_device_transfer
from tmd_policy.backends.lerobot.pi05_teacher import LeRobotPI05Teacher, PI05ConditionCache
from tmd_policy.backends.lerobot.smolvla_student import (
    LeRobotSmolVLAStudent,
    SmolVLAConditionCache,
    resolve_trainable_state_keys,
)
from tmd_policy.methods.discriminators import (
    CachedVLAFeatureDiscriminator,
    IntermediateFeatureDiscriminator,
)
from tmd_policy.methods.flow_objectives import (
    corrupt_rectified_flow,
    executable_coordinate_mask,
    sample_shifted_time,
    shifted_time_grid,
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


class _CapturedFirstOrderGradient(torch.autograd.Function):
    """Return an exact scalar value while replaying a captured input gradient."""

    @staticmethod
    def forward(ctx: Any, value: Tensor, gradient: Tensor, reported_loss: Tensor) -> Tensor:
        if value.shape != gradient.shape or value.device != gradient.device:
            raise ValueError("captured first-order gradient must match its input tensor")
        ctx.save_for_backward(gradient)
        return reported_loss.detach().clone()

    @staticmethod
    def backward(ctx: Any, output_gradient: Tensor) -> tuple[Tensor, None, None]:
        (gradient,) = ctx.saved_tensors
        multiplier = safe_device_transfer(output_gradient, gradient.device).to(gradient.dtype)
        return multiplier * gradient, None, None


def _stable_first_order_surrogate(
    loss: Tensor,
    value: Tensor,
    *,
    scales: tuple[float, ...] = (1.0, 2.0**-8, 2.0**-16, 2.0**-24),
) -> tuple[Tensor, float]:
    """Capture a finite first-order gradient without replaying an unstable graph.

    PI0.5's checkpoint-native BF16 suffix can produce a finite discriminator
    loss but overflow while differentiating that loss with respect to its
    action input. Scaling the scalar loss down and dividing the resulting FP32
    gradient by the same factor is mathematically identical at first order.
    The returned linear surrogate makes the training engine apply that captured
    gradient to the student without traversing the large suffix a second time.
    """

    if loss.numel() != 1 or not torch.isfinite(loss.detach()).all():
        raise RuntimeError("generator GAN loss contains NaN/Inf before backward")
    saw_connected = False
    saw_finite = False
    for scale in scales:
        raw_gradient = torch.autograd.grad(
            loss * scale,
            value,
            retain_graph=True,
            allow_unused=True,
        )[0]
        if raw_gradient is None:
            continue
        saw_connected = True
        gradient = raw_gradient.float() / scale
        if not torch.isfinite(gradient).all():
            continue
        saw_finite = True
        if torch.count_nonzero(gradient) == 0:
            continue
        gradient = gradient.to(value.dtype).detach()
        surrogate = _CapturedFirstOrderGradient.apply(
            value,
            gradient,
            loss.detach(),
        )
        if not torch.isfinite(surrogate.detach()).all():
            raise RuntimeError("captured generator GAN surrogate changed a finite loss to NaN/Inf")
        return surrogate, scale
    if not saw_connected:
        raise RuntimeError("generator GAN gradient is disconnected")
    if saw_finite:
        raise RuntimeError("generator GAN gradient is exactly zero")
    raise RuntimeError(
        "generator GAN gradient contains NaN/Inf at all safe backward scales"
    )


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
    """Direct DMD2-v outer-transition objective with VSD and feature GAN."""

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
        expected_training_mode = (
            "tmd_stage1_outer_transition"
            if preserve_student_trainability
            else "real_data_outer_transition"
        )
        if config.get("student_training_mode") != expected_training_mode:
            raise ValueError(
                f"student_training_mode must be {expected_training_mode!r} for this baseline"
            )
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
        self._validate_freezing_contract()

    def _validate_freezing_contract(self) -> None:
        teacher_parameters = list(self.teacher.policy.parameters())
        if not teacher_parameters or any(parameter.requires_grad for parameter in teacher_parameters):
            raise RuntimeError("PI0.5 teacher must be completely frozen")
        if any(parameter.grad is not None for parameter in teacher_parameters):
            raise RuntimeError("PI0.5 teacher contains stale gradients")
        if not any(parameter.requires_grad for parameter in self.fake_score.parameters()):
            raise RuntimeError("fake-score model contains no trainable parameters")
        if not any(parameter.requires_grad for parameter in self.discriminator.parameters()):
            raise RuntimeError("DMD2 discriminator contains no trainable parameters")

    @staticmethod
    def _require_module_gradient(module: nn.Module, label: str) -> None:
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError(f"{label} received no gradient")
        if not any(torch.count_nonzero(gradient) > 0 for gradient in gradients):
            raise RuntimeError(f"{label} received only zero gradients")
        if any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise RuntimeError(f"{label} gradient contains NaN/Inf")

    def validate_phase_gradients(self, phase: str) -> None:
        self._validate_freezing_contract()
        frozen_student = [
            parameter
            for parameter in self.student.parameters()
            if not parameter.requires_grad
        ]
        if any(parameter.grad is not None for parameter in frozen_student):
            raise RuntimeError("frozen SmolVLA parameters unexpectedly received gradients")
        if phase in {"guidance", "fake"}:
            self._require_module_gradient(self.fake_score, "fake-score model")
        if phase in {"guidance", "discriminator"}:
            self._require_module_gradient(self.discriminator, "DMD2 discriminator")
        if phase == "generator":
            self._require_module_gradient(self.student, "SmolVLA student")

    def to(self, *args: Any, **kwargs: Any) -> "DMD2FlowProgram":
        # This is a deliberately model-parallel program. Calling nn.Module.to
        # on the parent first would migrate the large PI0.5 fake suffix from
        # its designated GPU to the student GPU and then back. Besides doubling
        # peak traffic, BF16 peer-device round trips have corrupted expert
        # tensors on the supported stack. Move only student-owned state here.
        self.student.to(*args, **kwargs)
        self.bridge.to(*args, **kwargs)
        fake_devices = {parameter.device for parameter in self.fake_score.parameters()}
        if fake_devices != {self._fake_score_device}:
            raise RuntimeError(
                f"fake-score parameters left their designated device {self._fake_score_device}: "
                f"{sorted(map(str, fake_devices))}"
            )
        discriminator_devices = {
            parameter.device for parameter in self.discriminator.parameters()
        }
        if discriminator_devices != {self._discriminator_device}:
            self.discriminator.to(self._discriminator_device)
        return self

    def phase_schedule(self) -> tuple[str, ...]:
        ratio = int(self.dmd_config["fake_updates_per_generator"])
        if ratio < 1:
            raise ValueError("fake_updates_per_generator must be positive")
        if self.feature_source == "fake_score_features":
            return ("guidance",) * ratio + ("generator",)
        discriminator_ratio = int(self.dmd_config["discriminator_updates_per_generator"])
        if discriminator_ratio < 1:
            raise ValueError("discriminator_updates_per_generator must be positive")
        return ("fake",) * ratio + ("discriminator",) * discriminator_ratio + ("generator",)

    def backward_loss_scale(self, phase: str) -> float:
        # The combined guidance phase differentiates through the trainable BF16
        # PI0.5 suffix. A downscale avoids overflowing intermediate backward
        # values; the engine removes it before clipping and AdamW.
        if phase == "guidance" and self.fake_variant == "pi05_clone":
            return 2.0**-8
        return 1.0

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
        clean = self.student.policy.prepare_action(processed)
        source = torch.randn_like(clean)
        grid = shifted_time_grid(
            int(self.dmd_config["discrete_outer_steps"]),
            float(self.dmd_config["student_time_shift_gamma"]),
            device=self.student_device,
            dtype=torch.float32,
        )
        indices = torch.randint(1, grid.numel(), (condition.batch_size,), device=self.student_device)
        time = grid[indices]
        x_t = (1.0 - time[:, None, None]) * clean + time[:, None, None] * source

        def predict() -> Tensor:
            velocity = self.student.velocity(condition, x_t, time)
            return self._denoised_prediction(x_t, time, velocity)

        if requires_grad:
            return predict()
        with torch.no_grad():
            return predict().detach()

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
            prediction = self.fake_score(
                safe_device_transfer(x_t, self.student_device),
                safe_device_transfer(time, self.student_device),
                state,
                task,
            )
            return safe_device_transfer(prediction, x_t.device)
        target_device = next(self.fake_score.parameters()).device
        condition = condition if condition is not None else self._fake_condition(batch)
        prediction = self.fake_score(
            condition,
            safe_device_transfer(x_t, target_device),
            safe_device_transfer(time, target_device),
        )
        return safe_device_transfer(prediction, x_t.device)

    @staticmethod
    def _denoised_prediction(x_t: Tensor, time: Tensor, velocity: Tensor) -> Tensor:
        return x_t - time[:, None, None] * velocity

    def _real_actions_teacher(self, batch: dict[str, Any], valid: Tensor) -> Tensor:
        processed = self.student.preprocess_observation(batch)
        student_actions = self.student.policy.prepare_action(processed).detach()
        return self.bridge.student_to_teacher(student_actions, valid).values

    def _fake_loss(
        self,
        batch: dict[str, Any],
        generated: Tensor | None = None,
        *,
        teacher_condition: PI05ConditionCache | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        generated = (
            self._sample_student(batch, requires_grad=False)
            if generated is None
            else generated.detach()
        )
        valid_student = self._valid(batch, self.student_device)
        clean = self.bridge.student_to_teacher(generated, valid_student).values.detach()
        fake_device = self._fake_score_device
        clean = safe_device_transfer(clean, fake_device)
        valid = executable_coordinate_mask(
            safe_device_transfer(valid_student, fake_device), 32
        )
        time = self._sample_time("fake_score_time", clean.shape[0], fake_device)
        noise = torch.randn_like(clean)
        noised = corrupt_rectified_flow(clean, time, noise)
        target_velocity = noise - clean
        if self.fake_variant == "pi05_clone" and teacher_condition is None:
            teacher_condition = self._teacher_condition(batch)
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

    def _discriminator_loss(
        self,
        batch: dict[str, Any],
        generated: Tensor | None = None,
        *,
        teacher_condition: PI05ConditionCache | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        generated = (
            self._sample_student(batch, requires_grad=False)
            if generated is None
            else generated.detach()
        )
        valid_student = self._valid(batch, self.student_device)
        real = self._real_actions_teacher(batch, valid_student)
        fake = self.bridge.student_to_teacher(generated, valid_student).values.detach()
        if self.discriminator_variant == "pi05_intermediate_features":
            device = self._discriminator_device
            real = safe_device_transfer(real, device)
            fake = safe_device_transfer(fake, device)
            time = self._sample_time("gan_time", real.shape[0], device)
            real_noised = corrupt_rectified_flow(real, time, torch.randn_like(real))
            fake_noised = corrupt_rectified_flow(fake, time, torch.randn_like(fake))
            condition = (
                teacher_condition
                if teacher_condition is not None
                else self._teacher_condition(batch)
            )
            valid = safe_device_transfer(valid_student, device)
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
            real = safe_device_transfer(real, device)
            fake = safe_device_transfer(fake, device)
            real_noised = corrupt_rectified_flow(real, time, torch.randn_like(real))
            fake_noised = corrupt_rectified_flow(fake, time, torch.randn_like(fake))
            task = torch.as_tensor(batch["task_index"], device=device).long().flatten()
            valid = safe_device_transfer(valid_student, device)
            real_values = self.discriminator(real_noised, time, condition.condition_features, task, valid)
            fake_values = self.discriminator(fake_noised, time, condition.condition_features, task, valid)
            loss = F.softplus(-real_values).mean() + F.softplus(fake_values).mean()
        return loss, {
            "real_probability": float(real_values.sigmoid().mean().detach()),
            "fake_probability": float(fake_values.sigmoid().mean().detach()),
            "gan_noise_time": float(time.mean().detach()),
        }

    def _guidance_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        if self.feature_source != "fake_score_features":
            raise ValueError("combined guidance is only valid for fake-score discriminator features")
        # The frozen 3.45B-parameter PI0.5 prefix depends only on this
        # observation/language batch. Reuse its immutable KV cache for all
        # fake-score and classifier suffix queries in this loss.
        teacher_condition = self._teacher_condition(batch)
        generated = self._sample_student(batch, requires_grad=False).detach()
        fake_loss, fake_metrics = self._fake_loss(
            batch,
            generated,
            teacher_condition=teacher_condition,
        )
        if not torch.isfinite(fake_loss.detach()).all():
            raise RuntimeError("PI0.5 fake-score guidance loss contains NaN/Inf")
        discriminator_loss, discriminator_metrics = self._discriminator_loss(
            batch,
            generated,
            teacher_condition=teacher_condition,
        )
        if not torch.isfinite(discriminator_loss.detach()).all():
            raise RuntimeError("PI0.5 feature-discriminator guidance loss contains NaN/Inf")
        classifier_weight = float(self.dmd_config["guidance_classifier_weight"])
        loss = fake_loss + classifier_weight * discriminator_loss
        return loss, {
            **{f"fake_score/{name}": value for name, value in fake_metrics.items()},
            **{f"classifier/{name}": value for name, value in discriminator_metrics.items()},
            "fake_score_loss": float(fake_loss.detach()),
            "classifier_loss": float(discriminator_loss.detach()),
            "classifier_weight": classifier_weight,
        }

    def _distribution_matching_loss(
        self,
        batch: dict[str, Any],
        generated: Tensor,
        valid_student: Tensor,
        *,
        teacher_condition: PI05ConditionCache | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        teacher_generated = safe_device_transfer(
            self.bridge.student_to_teacher(generated, valid_student).values,
            self.teacher.device,
        )
        valid_coordinates = executable_coordinate_mask(
            safe_device_transfer(valid_student, self.teacher.device),
            teacher_generated.shape[-1],
        )
        time = self._sample_time("vsd_time", teacher_generated.shape[0], self.teacher.device)
        noised = corrupt_rectified_flow(teacher_generated, time, torch.randn_like(teacher_generated))
        condition = (
            teacher_condition
            if teacher_condition is not None
            else self._teacher_condition(batch)
        )
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
        self,
        batch: dict[str, Any],
        generated: Tensor,
        valid_student: Tensor,
        *,
        teacher_condition: PI05ConditionCache | None = None,
    ) -> Tensor:
        fake = self.bridge.student_to_teacher(generated, valid_student).values
        if self.discriminator_variant == "pi05_intermediate_features":
            device = self._discriminator_device
            fake = safe_device_transfer(fake, device)
            time = self._sample_time("gan_time", fake.shape[0], device)
            noised = corrupt_rectified_flow(fake, time, torch.randn_like(fake))
            condition = (
                teacher_condition
                if teacher_condition is not None
                else self._teacher_condition(batch)
            )
            with _frozen_parameters(self.discriminator):
                logits = self._paper_layer_logits(
                    condition,
                    noised,
                    time,
                    safe_device_transfer(valid_student, device),
                    require_input_grad=True,
                )
                loss = torch.stack([F.softplus(-value).mean() for value in logits.values()]).mean()
        else:
            condition = self._student_condition(batch)
            device = self._discriminator_device
            fake = safe_device_transfer(fake, device)
            time = self._sample_time("gan_time", fake.shape[0], device)
            noised = corrupt_rectified_flow(fake, time, torch.randn_like(fake))
            task = torch.as_tensor(batch["task_index"], device=device).long().flatten()
            with _frozen_parameters(self.discriminator):
                logits = self.discriminator(
                    noised,
                    time,
                    condition.condition_features,
                    task,
                    safe_device_transfer(valid_student, device),
                )
                loss = F.softplus(-logits).mean()
        loss, backward_scale = _stable_first_order_surrogate(loss, generated)
        self._last_gan_backward_scale = backward_scale
        if any(parameter.grad is not None for parameter in self.teacher.policy.parameters()):
            raise RuntimeError("frozen PI0.5 teacher unexpectedly received gradients")
        return loss

    def _generator_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        # VSD and feature-GAN queries share the same frozen PI0.5 condition.
        teacher_condition = self._teacher_condition(batch)
        generated = self._sample_student(batch, requires_grad=True)
        valid = self._valid(batch, self.student_device)
        vsd, metrics = self._distribution_matching_loss(
            batch,
            generated,
            valid,
            teacher_condition=teacher_condition,
        )
        gan = self._generator_gan_loss(
            batch,
            generated,
            valid,
            teacher_condition=teacher_condition,
        )
        total = vsd + float(self.dmd_config["gan_weight"]) * gan
        return total, {
            **metrics,
            "distribution_matching": float(vsd.detach()),
            "gan": float(gan.detach()),
            "gan_backward_scale": float(getattr(self, "_last_gan_backward_scale", 1.0)),
        }

    def loss(self, batch: dict[str, Any], phase: str) -> tuple[Tensor, dict[str, float]]:
        if phase == "guidance":
            return self._guidance_loss(batch)
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
            return {
                "guidance": torch.optim.AdamW(
                    [
                        {
                            "params": fake_parameters,
                            "lr": float(self.dmd_config["fake_score_learning_rate"]),
                        },
                        {
                            "params": discriminator_parameters,
                            "lr": float(self.dmd_config["discriminator_learning_rate"]),
                        },
                    ],
                    **common,
                ),
                "generator": torch.optim.AdamW(
                    [parameter for parameter in self.student.parameters() if parameter.requires_grad],
                    lr=float(self.dmd_config["generator_learning_rate"]),
                    **common,
                ),
            }
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

    def inference_state_dict(self) -> dict[str, Tensor]:
        """Save only the trained student delta on top of the immutable hub model."""

        student_state = self.student.state_dict()
        selected = resolve_trainable_state_keys(self.student)
        return {f"student.{name}": student_state[name].detach().cpu() for name in selected}

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
            "sampler_identity": (
                "DMD2-v real-data shifted outer transition during training; "
                "checkpoint-grid deterministic Euler integration during evaluation"
            ),
            "guidance_objective": (
                "fake_score_loss + guidance_classifier_weight * discriminator_loss"
                if self.feature_source == "fake_score_features"
                else "disjoint fake-score and discriminator update phases"
            ),
            "resource_model": self.dmd_config.get("resource_estimate", {}),
        }


__all__ = ["DMD2FlowProgram"]
