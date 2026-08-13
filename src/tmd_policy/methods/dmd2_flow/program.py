"""Paper-feature DMD2-flow with PI0.5 fake score, TTUR, VSD, and no SFT."""

from __future__ import annotations

import math
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


def _cosine(left: Tensor, right: Tensor) -> Tensor:
    return F.cosine_similarity(left.detach().float().flatten(1), right.detach().float().flatten(1))


def _time_binned_metrics(
    prefix: str,
    time: Tensor,
    values: Tensor,
    *,
    bins: int = 5,
) -> dict[str, float]:
    if time.ndim != 1 or values.shape != time.shape:
        raise ValueError("time-binned diagnostics require [B] times and values")
    indices = torch.clamp((time.detach().float() * bins).long(), max=bins - 1)
    result: dict[str, float] = {}
    for index in range(bins):
        selected = indices == index
        result[f"{prefix}/bin_{index}_fraction"] = float(selected.float().mean())
        if torch.any(selected):
            result[f"{prefix}/bin_{index}_value"] = float(values.detach().float()[selected].mean())
    return result


def _distribution_metrics(prefix: str, values: Tensor) -> dict[str, float]:
    flat = values.detach().float().flatten()
    quantiles = torch.quantile(flat, torch.tensor([0.1, 0.5, 0.9], device=flat.device))
    return {
        f"{prefix}/mean": float(flat.mean()),
        f"{prefix}/std": float(flat.std(unbiased=False)),
        f"{prefix}/p10": float(quantiles[0]),
        f"{prefix}/p50": float(quantiles[1]),
        f"{prefix}/p90": float(quantiles[2]),
    }


def _binary_auc(real_logits: Tensor, fake_logits: Tensor) -> float:
    real = real_logits.detach().float().flatten()
    fake = fake_logits.detach().float().flatten()
    if real.numel() == 0 or fake.numel() == 0:
        return float("nan")
    comparisons = real[:, None] - fake[None, :]
    return float(((comparisons > 0).float() + 0.5 * (comparisons == 0).float()).mean())


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
) -> tuple[Tensor, float, Tensor]:
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
        return surrogate, scale, gradient
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
    """DMD2 backward simulation with VSD, feature GAN, and no data regression."""

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
            else "backward_simulation_denoise_renoise"
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
        self._step_context: dict[str, Any] = {
            "global_step": 0,
            "phase": "initialization",
            "phase_occurrence": 0,
            "diagnostics_enabled": True,
        }
        self._student_replay: Any | None = None
        self._last_backward_metrics: dict[str, float] = {}
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

    @staticmethod
    def _gradient_group_metrics(
        prefix: str,
        parameters: list[Tensor],
    ) -> dict[str, float]:
        gradients = [parameter.grad.detach().float() for parameter in parameters if parameter.grad is not None]
        if not gradients:
            return {f"gradient/{prefix}/norm": 0.0, f"gradient/{prefix}/nonfinite_fraction": 0.0}
        squared = torch.stack([gradient.square().sum() for gradient in gradients]).sum()
        nonfinite = torch.stack([(~torch.isfinite(gradient)).sum() for gradient in gradients]).sum()
        elements = sum(gradient.numel() for gradient in gradients)
        return {
            f"gradient/{prefix}/norm": float(squared.sqrt()),
            f"gradient/{prefix}/nonfinite_fraction": float(nonfinite.float() / max(1, elements)),
        }

    def gradient_diagnostics(self, phase: str) -> dict[str, float]:
        if not bool(self._step_context.get("diagnostics_enabled", False)):
            return {}
        metrics: dict[str, float] = {}
        if phase in {"guidance", "fake"}:
            metrics.update(
                self._gradient_group_metrics(
                    "fake_pi05_suffix",
                    [parameter for parameter in self.fake_score.parameters() if parameter.requires_grad],
                )
            )
        if phase in {"guidance", "discriminator"}:
            metrics.update(
                self._gradient_group_metrics(
                    "discriminator_heads",
                    [parameter for parameter in self.discriminator.parameters() if parameter.requires_grad],
                )
            )
            heads = getattr(self.discriminator, "heads", {})
            for layer, head in heads.items():
                metrics.update(
                    self._gradient_group_metrics(
                        f"discriminator_layer_{layer}", list(head.parameters())
                    )
                )
        if phase == "generator":
            named = dict(self.student.named_parameters())
            groups = {
                "student_state_proj": ("state_proj",),
                "student_action_in_proj": ("action_in_proj",),
                "student_time_mlp": ("action_time_mlp_in", "action_time_mlp_out"),
                "student_action_out_proj": ("action_out_proj",),
            }
            for label, fragments in groups.items():
                parameters = [
                    parameter
                    for name, parameter in named.items()
                    if parameter.requires_grad and any(fragment in name for fragment in fragments)
                ]
                metrics.update(self._gradient_group_metrics(label, parameters))
        return metrics

    def optimizer_step_diagnostics(
        self,
        phase: str,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Compute the exact AdamW fake-parameter update ratio without cloning it."""

        if phase not in {"guidance", "fake"} or not bool(
            self._step_context.get("diagnostics_enabled", False)
        ):
            return {}
        fake_ids = {id(parameter) for parameter in self.fake_score.parameters() if parameter.requires_grad}
        delta_squares: list[Tensor] = []
        before_squares: list[Tensor] = []
        for group in optimizer.param_groups:
            beta1, beta2 = group["betas"]
            learning_rate = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            decay = 1.0 - learning_rate * weight_decay
            for parameter in group["params"]:
                if id(parameter) not in fake_ids or parameter not in optimizer.state:
                    continue
                state = optimizer.state[parameter]
                if "exp_avg" not in state or "exp_avg_sq" not in state:
                    continue
                step = int(torch.as_tensor(state["step"]).item())
                variance = state.get("max_exp_avg_sq", state["exp_avg_sq"])
                denominator = variance.float().sqrt() / math.sqrt(1.0 - beta2**step)
                denominator = denominator + float(group["eps"])
                adam_update = (
                    learning_rate
                    / (1.0 - beta1**step)
                    * state["exp_avg"].float()
                    / denominator
                )
                before = (parameter.detach().float() + adam_update) / decay
                delta = parameter.detach().float() - before
                delta_squares.append(delta.square().sum())
                before_squares.append(before.square().sum())
        if not delta_squares:
            return {"fake_score/parameter_update_ratio": 0.0}
        delta_norm = torch.stack(delta_squares).sum().sqrt()
        parameter_norm = torch.stack(before_squares).sum().sqrt().clamp_min(1.0e-30)
        return {"fake_score/parameter_update_ratio": float(delta_norm / parameter_norm)}

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

    def set_step_context(self, **values: Any) -> None:
        self._step_context = dict(values)

    def set_student_replay(self, replay: Any | None) -> None:
        self._student_replay = replay

    def prepare_phase_batch(self, batch: dict[str, Any], phase: str) -> dict[str, Any]:
        """Use replay observations for generator objectives, never GAN real updates."""

        if phase != "generator" or self._student_replay is None:
            return batch
        replay_batch = self._student_replay.sample_like(batch)
        return batch if replay_batch is None else replay_batch

    @staticmethod
    def _tensor_metrics(prefix: str, value: Tensor) -> dict[str, float]:
        detached = value.detach().float()
        return {
            f"{prefix}/mean": float(detached.mean()),
            f"{prefix}/std": float(detached.std(unbiased=False)),
            f"{prefix}/norm": float(
                torch.linalg.vector_norm(detached.flatten(1), dim=1).mean()
            ),
        }

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

    def _sample_student(
        self,
        batch: dict[str, Any],
        *,
        requires_grad: bool,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Simulate the DMD2 inference path up to a random denoising step.

        The batch shares one random step index, as in the released DMD2 code.
        Prefix steps start from pure Gaussian noise and alternate clean
        prediction with independent re-noising. Prefix computation is stopped;
        only the selected clean prediction carries a student gradient.
        """

        processed = self.student.preprocess_observation(batch)
        if requires_grad:
            condition = self.student.encode_condition(processed)
        else:
            with torch.no_grad():
                condition = self.student.encode_condition(processed)
        shape = (
            condition.batch_size,
            self.student.chunk_size,
            self.student.internal_action_dim,
        )
        source = torch.randn(
            shape,
            device=self.student_device,
            dtype=torch.float32,
            generator=generator,
        )
        steps = int(self.dmd_config["discrete_outer_steps"])
        grid = shifted_time_grid(
            steps,
            float(self.dmd_config["student_time_shift_gamma"]),
            device=self.student_device,
            dtype=torch.float32,
            descending=True,
        )
        selected = int(
            torch.randint(
                0,
                steps,
                (1,),
                device=self.student_device,
                generator=generator,
            ).item()
        )
        value = source
        metrics = {
            "backward/selected_step_index": float(selected),
            "backward/selected_time": float(grid[selected]),
            "backward/pure_noise_step_fraction": float(selected == 0),
            **self._tensor_metrics("backward/source_noise", source),
        }
        for index in range(selected):
            current = grid[index].expand(condition.batch_size)
            with torch.no_grad():
                clean_prefix = self._denoised_prediction(
                    value,
                    current,
                    self.student.velocity(condition, value, current),
                ).detach()
            target = grid[index + 1]
            fresh_noise = torch.randn(
                shape,
                device=self.student_device,
                dtype=torch.float32,
                generator=generator,
            )
            next_value = (1.0 - target) * clean_prefix + target * fresh_noise
            metrics.update(self._tensor_metrics(f"backward/x_t_prefix_{index}", value))
            metrics[f"backward/transition_{index}_norm"] = float(
                torch.linalg.vector_norm((next_value - value).flatten(1), dim=1).mean()
            )
            value = next_value

        time = grid[selected].expand(condition.batch_size)

        def predict() -> Tensor:
            velocity = self.student.velocity(condition, value, time)
            return self._denoised_prediction(value, time, velocity)

        metrics.update(self._tensor_metrics(f"backward/x_t_prefix_{selected}", value))
        for index in range(steps):
            metrics[f"backward/t_bin_{index}_fraction"] = float(index == selected)
        if requires_grad:
            generated = predict()
        else:
            with torch.no_grad():
                generated = predict().detach()
        metrics.update(self._tensor_metrics("backward/generated_clean", generated))
        self._last_backward_metrics = metrics
        return generated

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
        return x_t.float() - time.float()[:, None, None] * velocity.float()

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
        per_sample = _masked_mean((prediction.float() - target_velocity.float()).square(), valid)
        loss = per_sample.mean()
        metrics = {
            "velocity_mse": float(loss.detach()),
            **_time_binned_metrics("timestep_fake_mse", time, per_sample),
        }
        diagnostics = bool(self._step_context.get("diagnostics_enabled", False))
        if diagnostics and self.fake_variant == "pi05_clone":
            if teacher_condition is None:
                teacher_condition = self._teacher_condition(batch)
            with torch.no_grad():
                teacher_velocity = self.teacher.velocity(
                    teacher_condition,
                    safe_device_transfer(noised.detach().float(), self.teacher.device),
                    safe_device_transfer(time.detach().float(), self.teacher.device),
                )
            teacher_velocity = safe_device_transfer(teacher_velocity, prediction.device)
            teacher_clean = self._denoised_prediction(noised, time, teacher_velocity)
            fake_clean = self._denoised_prediction(noised, time, prediction)
            metrics.update(
                {
                    "teacher_fake_velocity_cosine": float(
                        _cosine(teacher_velocity, prediction).mean()
                    ),
                    "teacher_fake_clean_prediction_l1": float(
                        _masked_mean((teacher_clean - fake_clean).abs(), valid).mean()
                    ),
                    "fixed_student_sample_tracking_lag": float(
                        _masked_mean((teacher_clean - fake_clean).abs(), valid).mean()
                    ),
                }
            )
        return loss, metrics

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
            real_values = torch.stack(list(real_logits.values())).mean(dim=0)
            fake_values = torch.stack(list(fake_logits.values())).mean(dim=0)
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
            real_logits = {}
            fake_logits = {}
        real_correct = real_values > 0
        fake_correct = fake_values < 0
        accuracy = torch.cat((real_correct, fake_correct)).float().mean()
        per_example_accuracy = 0.5 * (real_correct.float() + fake_correct.float())
        metrics = {
            **_distribution_metrics("real_logits", real_values),
            **_distribution_metrics("fake_logits", fake_values),
            "real_probability": float(real_values.sigmoid().mean().detach()),
            "fake_probability": float(fake_values.sigmoid().mean().detach()),
            "accuracy": float(accuracy),
            "auc": _binary_auc(real_values, fake_values),
            "gan_noise_time": float(time.mean().detach()),
            **_time_binned_metrics(
                "timestep_discriminator_accuracy", time, per_example_accuracy
            ),
        }
        for index in self.selected_layers:
            if index not in real_logits:
                continue
            layer_loss = F.softplus(-real_logits[index]).mean() + F.softplus(
                fake_logits[index]
            ).mean()
            metrics.update(
                {
                    f"layer_{index}/loss": float(layer_loss.detach()),
                    f"layer_{index}/real_logit": float(real_logits[index].mean().detach()),
                    f"layer_{index}/fake_logit": float(fake_logits[index].mean().detach()),
                }
            )
        return loss, metrics

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
            **getattr(self, "_last_backward_metrics", {}),
            **{f"fake_score/{name}": value for name, value in fake_metrics.items()},
            **{f"classifier/{name}": value for name, value in discriminator_metrics.items()},
            "fake_score_loss": float(fake_loss.detach()),
            "classifier_loss": float(discriminator_loss.detach()),
            "discriminator_loss": float(discriminator_loss.detach()),
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
        denominator = values["denominator"].detach().float()
        denominator_quantiles = torch.quantile(
            denominator,
            torch.tensor([0.5, 0.95], device=denominator.device),
        )
        metrics = {name: float(value.float().mean()) for name, value in values.items()}
        metrics.update(
            {
                "normalization_denominator_min": float(denominator.min()),
                "normalization_denominator_p50": float(denominator_quantiles[0]),
                "normalization_denominator_p95": float(denominator_quantiles[1]),
                "teacher_prediction_norm": float(
                    torch.linalg.vector_norm(teacher_prediction.float().flatten(1), dim=1).mean()
                ),
                "fake_prediction_norm": float(
                    torch.linalg.vector_norm(fake_prediction.float().flatten(1), dim=1).mean()
                ),
                "teacher_fake_cosine_similarity": float(
                    _cosine(teacher_prediction, fake_prediction).mean()
                ),
                "direction_norm": float(
                    torch.linalg.vector_norm(direction.float().flatten(1), dim=1).mean()
                ),
                "direction_max": float(direction.detach().float().abs().max()),
            }
        )
        return loss, metrics

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
        loss, backward_scale, action_gradient = _stable_first_order_surrogate(loss, generated)
        self._last_gan_backward_scale = backward_scale
        self._last_gan_action_gradient = action_gradient.detach().float()
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
        dmd_gradient = torch.autograd.grad(
            vsd,
            generated,
            retain_graph=True,
            allow_unused=False,
        )[0].detach().float()
        gan_gradient = getattr(self, "_last_gan_action_gradient", None)
        if gan_gradient is None:
            # Keep the helper independently testable when a custom GAN loss is
            # injected. The production path captures this gradient with
            # _stable_first_order_surrogate so the large PI0.5 graph is not
            # replayed a second time.
            gan_gradient = torch.autograd.grad(
                gan,
                generated,
                retain_graph=True,
                allow_unused=False,
            )[0].detach().float()
        gradient_cosine = F.cosine_similarity(
            dmd_gradient.flatten(1),
            gan_gradient.flatten(1),
        ).mean()
        total = vsd + float(self.dmd_config["gan_weight"]) * gan
        canonical = self.bridge.student_to_canonical(generated).detach().float()
        generated_metrics = self._tensor_metrics("generated/action", canonical)
        for index in range(canonical.shape[-1]):
            generated_metrics[f"generated/action_dim_{index}_mean"] = float(
                canonical[..., index].mean()
            )
            generated_metrics[f"generated/action_dim_{index}_std"] = float(
                canonical[..., index].std(unbiased=False)
            )
        adjacent_valid = valid[:, 1:] & valid[:, :-1]
        smoothness = torch.linalg.vector_norm(canonical[:, 1:] - canonical[:, :-1], dim=-1)
        generated_metrics["generated/chunk_smoothness"] = float(
            (smoothness * adjacent_valid.float()).sum()
            / adjacent_valid.float().sum().clamp_min(1.0)
        )
        generated_metrics["generated/valid_fraction"] = float(valid.float().mean())
        generated_metrics["generated/replay_state_fraction"] = float(
            bool(batch.get("_student_replay_batch", False))
        )
        return total, {
            **getattr(self, "_last_backward_metrics", {}),
            **metrics,
            **generated_metrics,
            "distribution_matching": float(vsd.detach()),
            "distribution_matching_loss": float(vsd.detach()),
            "gan": float(gan.detach()),
            "generator_gan_loss": float(gan.detach()),
            "generator_total_loss": float(total.detach()),
            "vsd_gradient_norm": float(
                torch.linalg.vector_norm(dmd_gradient.flatten(1), dim=1).mean()
            ),
            "generated_action_gan_input_gradient_norm": float(
                torch.linalg.vector_norm(gan_gradient.flatten(1), dim=1).mean()
            ),
            "dmd_gan_gradient_cosine_similarity": float(gradient_cosine),
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
        devices = sorted(
            {
                device.index
                for device in (self.student_device, self.teacher.device, self._fake_score_device)
                if device.type == "cuda" and device.index is not None
            }
        )
        previous_context = self._step_context
        self._step_context = {**previous_context, "diagnostics_enabled": True, "phase": "validation"}
        try:
            # Validation loader order, observation batches, and all random
            # draws are fixed across calls. fork_rng prevents this probe from
            # perturbing the subsequent training stream.
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(0xD2D2)
                for device_index in devices:
                    with torch.cuda.device(device_index):
                        torch.cuda.manual_seed(0xD2D2)
                generator = torch.Generator(device=self.student_device).manual_seed(0xD2D2)
                generated = self._sample_student(
                    batch,
                    requires_grad=False,
                    generator=generator,
                )
                teacher_condition = self._teacher_condition(batch)
                fake_loss, fake_metrics = self._fake_loss(
                    batch,
                    generated,
                    teacher_condition=teacher_condition,
                )
                valid = self._valid(batch, self.student_device)
                dmd_loss, dmd_metrics = self._distribution_matching_loss(
                    batch,
                    generated,
                    valid,
                    teacher_condition=teacher_condition,
                )
        finally:
            self._step_context = previous_context
        return fake_loss, {
            **{f"fixed_probe/fake_score/{name}": value for name, value in fake_metrics.items()},
            **{f"fixed_probe/dmd/{name}": value for name, value in dmd_metrics.items()},
            **{f"fixed_probe/{name}": value for name, value in self._last_backward_metrics.items()},
            "fixed_probe/fake_score_loss": float(fake_loss.detach()),
            "fixed_probe/generator_loss": float(dmd_loss.detach()),
            "fixed_probe/generator_distribution_matching_loss": float(dmd_loss.detach()),
            "fixed_probe/teacher_prediction_norm": float(
                dmd_metrics["teacher_prediction_norm"]
            ),
            "fixed_probe/teacher_fake_clean_l1": float(
                fake_metrics.get(
                    "teacher_fake_clean_prediction_l1", dmd_metrics["difference_l1"]
                )
            ),
        }

    def make_optimizers(self, training: dict[str, Any]) -> dict[str, torch.optim.Optimizer]:
        common = {
            "betas": tuple(training.get("betas", [0.9, 0.95])),
            "weight_decay": float(training["weight_decay"]),
            "eps": float(training.get("epsilon", 1.0e-8)),
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
                "DMD2 backward simulation during training and DMD2 denoise-renoise "
                "sampling during evaluation; both start from independent Gaussian noise"
            ),
            "student_state_replay": self.dmd_config.get("student_rollout_replay"),
            "guidance_objective": (
                "fake_score_loss + guidance_classifier_weight * discriminator_loss"
                if self.feature_source == "fake_score_features"
                else "disjoint fake-score and discriminator update phases"
            ),
            "resource_model": self.dmd_config.get("resource_estimate", {}),
        }


__all__ = ["DMD2FlowProgram"]
