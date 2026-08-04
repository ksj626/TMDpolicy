"""Online PI0.5→SmolVLA DMD2-flow with shared student sampler."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tmd_policy.backends.action_coordinates import ActionCoordinateBridge
from tmd_policy.backends.lerobot.pi05_teacher import LeRobotPI05Teacher, PI05ConditionCache
from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent
from tmd_policy.training.engine import TrainingProgram

from .fake_scores import PI05CloneFakeScore, SmolVLACloneFakeScore
from .networks import ActionChunkDiscriminator, ActionScoreTransformer


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    mask = valid.unsqueeze(-1).to(values.dtype)
    return (values * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) * values.shape[-1]).clamp_min(1)


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
    """Constructs teacher, student, fake score, GAN critic, data loss, and TTUR."""

    def __init__(
        self,
        *,
        student: LeRobotSmolVLAStudent,
        teacher: LeRobotPI05Teacher,
        bridge: ActionCoordinateBridge,
        config: dict[str, Any],
        fake_score: nn.Module | None = None,
    ) -> None:
        super().__init__()
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
            self.fake_classification = "lightweight action-score adaptation; not paper-faithful"
        elif self.fake_variant in {"pi05_clone", "smolvla_clone"}:
            if fake_score is None:
                raise ValueError(f"{self.fake_variant} requires a constructed clone backend")
            expected = PI05CloneFakeScore if self.fake_variant == "pi05_clone" else SmolVLACloneFakeScore
            if not isinstance(fake_score, expected):
                raise TypeError(f"{self.fake_variant} requires {expected.__name__}")
            self.fake_score = fake_score
            self.fake_classification = fake_score.classification
        else:
            raise ValueError(f"unknown fake-score variant: {self.fake_variant}")
        disc = config["discriminator"]
        self.discriminator = ActionChunkDiscriminator(
            num_tasks=int(disc["num_tasks"]),
            model_dim=int(disc["model_dim"]),
            layers=int(disc["layers"]),
            heads=int(disc["heads"]),
        )
        self.student.configure_trainable(str(config["student_fine_tuning"]))
        # `TrainingProgram.run_training` places ordinary modules on the student
        # device. Preserve the explicitly configured fake-score placement so a
        # PI0.5/SmolVLA clone can remain on its own GPU.
        self._fake_score_device = next(self.fake_score.parameters()).device

    def to(self, *args: Any, **kwargs: Any) -> "DMD2FlowProgram":
        super().to(*args, **kwargs)
        self.fake_score.to(self._fake_score_device)
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

    def _sample_student(self, batch: dict[str, Any], *, requires_grad: bool) -> Tensor:
        processed = self.student.preprocess_observation(batch)
        condition = self.student.encode_condition(processed)
        noise = torch.randn(
            condition.batch_size,
            50,
            32,
            device=self.student_device,
            dtype=torch.float32,
        )
        if requires_grad:
            return self.student.sample(
                condition, noise, int(self.dmd_config["generation_steps"])
            )
        with torch.no_grad():
            return self.student.sample(
                condition, noise, int(self.dmd_config["generation_steps"])
            ).detach()

    def _teacher_condition(self, batch: dict[str, Any]) -> PI05ConditionCache:
        return self.teacher.encode_condition(self.teacher.preprocess_observation(batch))

    def _fake_condition(self, batch: dict[str, Any]) -> Any:
        if self.fake_variant == "lightweight":
            return None
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
            return self.fake_score(x_t.to(self.student_device), time.to(self.student_device), state, task).to(x_t.device)
        condition = condition if condition is not None else self._fake_condition(batch)
        target_device = next(self.fake_score.parameters()).device
        return self.fake_score(condition, x_t.to(target_device), time.to(target_device)).to(x_t.device)

    def _fake_score_value(
        self, batch: dict[str, Any], x_t: Tensor, time: Tensor, condition: Any | None = None
    ) -> Tensor:
        velocity = self._fake_velocity(batch, x_t, time, condition)
        safe = time.clamp_min(float(self.dmd_config["minimum_score_time"]))[:, None, None]
        return -(x_t + (1.0 - safe) * velocity) / safe

    def _fake_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        student_generated = self._sample_student(batch, requires_grad=False)
        valid_student = self._valid(batch, self.student_device)
        teacher_generated = self.bridge.student_to_teacher(student_generated, valid_student).values.detach()
        fake_device = next(self.fake_score.parameters()).device
        teacher_generated = teacher_generated.to(fake_device)
        valid = valid_student.to(fake_device)
        batch_size = teacher_generated.shape[0]
        time = torch.empty(batch_size, device=fake_device).uniform_(
            float(self.dmd_config["minimum_score_time"]),
            float(self.dmd_config["maximum_score_time"]),
        )
        noise = torch.randn_like(teacher_generated)
        x_t = (1.0 - time[:, None, None]) * teacher_generated + time[:, None, None] * noise
        target_velocity = noise - teacher_generated
        prediction = self._fake_velocity(batch, x_t, time)
        # Only seven coordinates correspond to executable LIBERO actions. The
        # remaining PI0.5 dimensions are padding and must not train the fake
        # score or influence its normalization.
        error = (prediction[..., :7] - target_velocity[..., :7]).square()
        loss = _masked_mean(error, valid).mean()
        return loss, {"velocity_mse": float(loss.detach())}

    def _discriminator_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        generated = self._sample_student(batch, requires_grad=False)[..., :7]
        processed = self.student.preprocess_observation(batch)
        real = self.student.policy.prepare_action(processed)[..., :7].detach()
        valid = self._valid(batch, self.student_device)
        state, task = self._raw_state_task(batch)
        real_logits = self.discriminator(real, state, task, valid)
        fake_logits = self.discriminator(generated.detach(), state, task, valid)
        loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        loss = loss + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return loss, {
            "real_probability": float(real_logits.sigmoid().mean().detach()),
            "fake_probability": float(fake_logits.sigmoid().mean().detach()),
        }

    def _distribution_matching_loss(
        self, batch: dict[str, Any], generated: Tensor, valid_student: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        teacher_generated = self.bridge.student_to_teacher(generated, valid_student).values.to(self.teacher.device)
        valid = valid_student.to(self.teacher.device)
        batch_size = teacher_generated.shape[0]
        time = torch.empty(batch_size, device=self.teacher.device).uniform_(
            float(self.dmd_config["minimum_score_time"]),
            float(self.dmd_config["maximum_score_time"]),
        )
        noise = torch.randn_like(teacher_generated)
        x_t = (1.0 - time[:, None, None]) * teacher_generated + time[:, None, None] * noise
        teacher_condition = self._teacher_condition(batch)
        real_score = self.teacher.score(teacher_condition, x_t.detach(), time).detach()
        with torch.no_grad():
            fake_score = self._fake_score_value(batch, x_t.detach(), time).detach()
        dimension_valid = self.bridge.teacher_dimension_valid.to(self.teacher.device)
        coordinates = valid.unsqueeze(-1) & dimension_valid[None, None]
        normalizer = _masked_mean(real_score[..., :7].abs(), valid).clamp_min(1e-6)[:, None, None]
        gradient = (fake_score - real_score) / normalizer
        surrogate = teacher_generated * gradient.detach() * coordinates.to(teacher_generated.dtype)
        loss = surrogate.sum() / coordinates.sum().clamp_min(1)
        return loss, {
            "teacher_score_abs": float(real_score.abs()[coordinates].mean()),
            "fake_score_abs": float(fake_score.abs()[coordinates].mean()),
            "coordinate_gradient_abs": float(gradient.abs()[coordinates].mean()),
        }

    def _generator_loss(self, batch: dict[str, Any]) -> tuple[Tensor, dict[str, float]]:
        generated = self._sample_student(batch, requires_grad=True)
        valid = self._valid(batch, self.student_device)
        dmd, metrics = self._distribution_matching_loss(batch, generated, valid)
        state, task = self._raw_state_task(batch)
        with _frozen_parameters(self.discriminator):
            gan_logits = self.discriminator(generated[..., :7], state, task, valid)
        gan = F.binary_cross_entropy_with_logits(gan_logits, torch.ones_like(gan_logits))
        data = self.student.flow_matching_loss(batch)
        total = dmd + float(self.dmd_config["gan_weight"]) * gan + float(
            self.dmd_config["data_weight"]
        ) * data
        return total, {
            **metrics,
            "distribution_matching": float(dmd.detach()),
            "gan": float(gan.detach()),
            "data": float(data.detach()),
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
        common = {"betas": tuple(training.get("betas", [0.9, 0.95])), "weight_decay": float(training["weight_decay"])}
        student_parameters = [parameter for parameter in self.student.parameters() if parameter.requires_grad]
        fake_parameters = [parameter for parameter in self.fake_score.parameters() if parameter.requires_grad]
        discriminator_parameters = [parameter for parameter in self.discriminator.parameters() if parameter.requires_grad]
        return {
            "fake": torch.optim.AdamW(fake_parameters, lr=float(self.dmd_config["fake_score_learning_rate"]), **common),
            "discriminator": torch.optim.AdamW(
                discriminator_parameters,
                lr=float(self.dmd_config["discriminator_learning_rate"]),
                **common,
            ),
            "generator": torch.optim.AdamW(
                student_parameters,
                lr=float(self.dmd_config["generator_learning_rate"]),
                **common,
            ),
        }

    def extra_provenance(self) -> dict[str, Any]:
        return {
            "dmd2": self.dmd_config,
            "fake_score_classification": self.fake_classification,
            "teacher_device": str(self.teacher.device),
            "sampler_identity": "LeRobotSmolVLAStudent.sample for training simulation and inference",
            "resource_model": self.dmd_config.get("resource_estimate", {}),
        }


__all__ = ["DMD2FlowProgram"]
