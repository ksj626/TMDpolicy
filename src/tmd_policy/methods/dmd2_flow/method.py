from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tmd_policy.common.capabilities import Capability, CapabilitySet
from tmd_policy.common.checkpointing import load_method_checkpoint, save_method_checkpoint
from tmd_policy.common.density import RectifiedFlowSchedule
from tmd_policy.methods.base import DryRunReport, ResearchMethod

from .losses import (
    ConditionalActionGAN,
    discriminator_loss,
    dmd2_distribution_matching_loss,
    fake_score_loss,
    generator_gan_loss,
)


@dataclass(frozen=True)
class DMD2Config:
    fake_updates_per_generator: int = 5
    generator_learning_rate: float = 1e-5
    fake_score_learning_rate: float = 1e-5
    discriminator_learning_rate: float = 1e-5
    gan_weight: float = 3e-3
    minimum_score_time: float = 1e-3
    maximum_score_time: float = 0.999
    generation_schedule: tuple[float, ...] = (0.999, 0.749, 0.499, 0.249)

    def __post_init__(self) -> None:
        if self.fake_updates_per_generator < 1:
            raise ValueError("DMD2 TTUR requires at least one fake-score update")
        if min(
            self.generator_learning_rate,
            self.fake_score_learning_rate,
            self.discriminator_learning_rate,
        ) <= 0 or self.gan_weight < 0:
            raise ValueError("invalid DMD2 learning rate or GAN weight")
        if not self.generation_schedule or any(
            left <= right for left, right in zip(self.generation_schedule[:-1], self.generation_schedule[1:])
        ):
            raise ValueError("DMD2 generation schedule must be strictly descending")


def simulate_multistep_inputs(
    generator: Callable[[Tensor, Tensor, Tensor], Tensor],
    initial_noise: Tensor,
    condition: Tensor,
    schedule: tuple[float, ...],
    reinjection_noises: tuple[Tensor, ...],
) -> tuple[Tensor, ...]:
    if len(reinjection_noises) != max(0, len(schedule) - 1):
        raise ValueError("one independent reinjection noise is required between generator steps")
    current = initial_noise
    outputs: list[Tensor] = []
    for index, time_value in enumerate(schedule):
        time = torch.full((current.shape[0],), time_value, device=current.device, dtype=current.dtype)
        clean = generator(current, time, condition)
        outputs.append(current)
        if index < len(reinjection_noises):
            next_time = schedule[index + 1]
            current = ((1 - next_time) * clean + next_time * reinjection_noises[index]).detach()
    return tuple(outputs)


class DMD2FlowMethod(ResearchMethod):
    name = "dmd2_flow"
    classification = "paper-faithful action-flow port"

    def __init__(
        self,
        *,
        generator: nn.Module,
        fake_score: nn.Module,
        discriminator: ConditionalActionGAN,
        teacher_velocity: Callable[[Tensor, Tensor, Tensor], Tensor],
        config: DMD2Config,
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
        self.generator = generator
        self.fake_score = fake_score
        self.discriminator = discriminator
        self.teacher_velocity = teacher_velocity
        self.config = config
        self.schedule = RectifiedFlowSchedule(config.minimum_score_time, config.maximum_score_time)
        self.optimizers = {
            "generator": torch.optim.AdamW(generator.parameters(), lr=config.generator_learning_rate),
            "fake_score": torch.optim.AdamW(fake_score.parameters(), lr=config.fake_score_learning_rate),
            "discriminator": torch.optim.AdamW(
                discriminator.parameters(), lr=config.discriminator_learning_rate
            ),
        }
        self.schedulers = {
            name: torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
            for name, optimizer in self.optimizers.items()
        }
        self.counters = {"generator": 0, "fake_score": 0, "discriminator": 0}
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
        self.config.__post_init__()

    @staticmethod
    def _set_trainable(module: nn.Module, value: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(value)

    def training_step(self, batch: Mapping[str, Any]) -> Mapping[str, Tensor]:
        condition: Tensor = batch["condition"]
        valid_mask: Tensor = batch["valid_mask"]
        if tuple(batch["condition_task_uids"]) != tuple(batch["gan_real_task_uids"]):
            raise ValueError("DMD2 GAN real chunks must match generated conditioning tasks exactly")
        simulated_inputs = simulate_multistep_inputs(
            self.generator,
            batch["simulation_initial_noise"],
            condition,
            self.config.generation_schedule,
            tuple(batch["simulation_reinjection_noises"]),
        )
        simulation_index = int(batch["simulation_step_index"])
        generator_input = simulated_inputs[simulation_index]
        generator_time = torch.full(
            (generator_input.shape[0],),
            self.config.generation_schedule[simulation_index],
            device=generator_input.device,
            dtype=generator_input.dtype,
        )
        generated = self.generator(generator_input, generator_time, condition)
        fake_losses: list[Tensor] = []
        for _ in range(self.config.fake_updates_per_generator):
            self._set_trainable(self.generator, False)
            self._set_trainable(self.fake_score, True)
            self._set_trainable(self.discriminator, True)
            fake_noisy_actions = self.schedule.interpolate(
                generated.detach(), batch["fake_noise"], batch["fake_time"]
            )
            fake_prediction = self.fake_score(fake_noisy_actions, batch["fake_time"], condition)
            fake_loss = fake_score_loss(
                fake_prediction, batch["fake_noise"] - generated.detach(), valid_mask
            )
            fake_logits = self.discriminator(
                fake_noisy_actions.detach(), condition, valid_mask
            )
            real_logits = self.discriminator(batch["gan_real_noisy"], condition, valid_mask)
            discriminator_value = discriminator_loss(real_logits, fake_logits)
            guidance_loss = fake_loss + discriminator_value
            self.optimizers["fake_score"].zero_grad(set_to_none=True)
            self.optimizers["discriminator"].zero_grad(set_to_none=True)
            guidance_loss.backward()
            self.optimizers["fake_score"].step()
            self.optimizers["discriminator"].step()
            self.schedulers["fake_score"].step()
            self.schedulers["discriminator"].step()
            self.counters["fake_score"] += 1
            self.counters["discriminator"] += 1
            fake_losses.append(fake_loss.detach())
        self._set_trainable(self.generator, True)
        self._set_trainable(self.fake_score, False)
        self._set_trainable(self.discriminator, False)
        dm = dmd2_distribution_matching_loss(
            generated,
            time=batch["score_time"],
            corruption_noise=batch["score_noise"],
            valid_mask=valid_mask,
            real_velocity=lambda state, time: self.teacher_velocity(state, time, condition),
            fake_velocity=lambda state, time: self.fake_score(state, time, condition),
            schedule=self.schedule,
        )
        # Construct this from the live sample so GAN gradients reach the
        # generator rather than stopping at a caller-provided tensor.
        generator_gan_input = self.schedule.interpolate(
            generated, batch["gan_generator_noise"], batch["gan_time"]
        )
        generator_logits = self.discriminator(generator_gan_input, condition, valid_mask)
        gan_value = generator_gan_loss(generator_logits)
        generator_loss = dm["loss"] + self.config.gan_weight * gan_value
        self.optimizers["generator"].zero_grad(set_to_none=True)
        generator_loss.backward()
        self.optimizers["generator"].step()
        self.schedulers["generator"].step()
        self.counters["generator"] += 1
        return {
            "loss": generator_loss.detach(),
            "distribution_matching_loss": dm["loss"].detach(),
            "generator_gan_loss": gan_value.detach(),
            "fake_score_loss": torch.stack(fake_losses).mean(),
        }

    def sample_action_chunk(self, batch: Mapping[str, Any]) -> Tensor:
        current = batch["noise"]
        condition = batch["condition"]
        for time_value in self.config.generation_schedule:
            time = torch.full((current.shape[0],), time_value, device=current.device, dtype=current.dtype)
            current = self.generator(current, time, condition)
        return current

    def save_method_state(self, path: str | Path) -> Path:
        return save_method_checkpoint(
            path,
            method_name=self.name,
            models={
                "generator": self.generator,
                "fake_score": self.fake_score,
                "gan_discriminator": self.discriminator,
            },
            optimizers=self.optimizers,
            schedulers=self.schedulers,
            scaler=None,
            counters=self.counters,
            config=asdict(self.config),
            provenance={},
            trainable_names={
                "generator": [n for n, p in self.generator.named_parameters() if p.requires_grad],
                "fake_score": [n for n, _ in self.fake_score.named_parameters()],
                "gan_discriminator": [n for n, _ in self.discriminator.named_parameters()],
            },
        )

    def load_method_state(self, path: str | Path) -> Mapping[str, Any]:
        value = load_method_checkpoint(
            path,
            expected_method=self.name,
            models={
                "generator": self.generator,
                "fake_score": self.fake_score,
                "gan_discriminator": self.discriminator,
            },
            optimizers=self.optimizers,
            schedulers=self.schedulers,
        )
        self.counters = dict(value["counters"])
        return value

    def dry_run_report(self) -> DryRunReport:
        if self._dry_run is None:
            raise RuntimeError("dry-run context was not provided")
        return self._dry_run
