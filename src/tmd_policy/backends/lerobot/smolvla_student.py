"""SmolVLA baseline and DMD2 action-expert student backend."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from tmd_policy.methods.flow_objectives import shifted_time_grid
from tmd_policy.methods.denoise_renoise import denoise_renoise_sample

from .compatibility import (
    validate_smolvla_instance,
    verify_checkpoint_projection_loaded,
    verify_installed_lerobot,
)

FineTuningMode = Literal["action_expert", "head_only"]


def resolve_trainable_state_keys(student: nn.Module) -> tuple[str, ...]:
    """Map policy-relative trainable names to wrapper state-dict keys."""

    state_keys = set(student.state_dict())
    resolved = []
    for parameter_name in getattr(student, "trainable_parameter_names", ()):
        candidates = (str(parameter_name), f"policy.{parameter_name}")
        matches = [candidate for candidate in candidates if candidate in state_keys]
        if len(matches) != 1:
            raise RuntimeError(
                "cannot uniquely map trainable SmolVLA parameter to its wrapper state dict: "
                f"{parameter_name!r} -> {matches}"
            )
        resolved.append(matches[0])
    if not resolved:
        raise RuntimeError("SmolVLA exposes no trainable state-dict keys")
    return tuple(resolved)


@dataclass(frozen=True)
class SmolVLAConditionCache:
    prefix_pad_masks: Tensor
    past_key_values: Any
    batch_size: int
    prefix_length: int
    model_revision: str
    processor_revision: str
    dtype: str
    device: str


def _unique_parameters(modules: list[nn.Module]) -> set[int]:
    return {id(parameter) for module in modules for parameter in module.parameters()}


class LeRobotSmolVLAStudent(nn.Module):
    """Stable access to official SmolVLA loss, velocity, and sampling."""

    chunk_size = 50
    internal_action_dim = 32

    def __init__(
        self,
        policy: nn.Module,
        preprocessor: Any,
        postprocessor: Any,
        *,
        model_id: str,
        model_revision: str,
        processor_revision: str,
        device: str | torch.device,
    ) -> None:
        super().__init__()
        validate_smolvla_instance(policy)
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.model_id = model_id
        self.model_revision = model_revision
        self.processor_revision = processor_revision
        self.device = torch.device(device)
        self.trainable_parameter_names: tuple[str, ...] = ()
        if int(policy.config.chunk_size) != self.chunk_size or int(policy.config.max_action_dim) != self.internal_action_dim:
            raise ValueError("loaded SmolVLA checkpoint does not implement the required [50,32] flow")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str,
        processor_revision: str,
        device: str | torch.device = "cuda:0",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        expected_source_hashes: dict[str, str] | None = None,
    ) -> "LeRobotSmolVLAStudent":
        verify_installed_lerobot(expected_source_hashes=expected_source_hashes)
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla import SmolVLAPolicy

        snapshot = snapshot_download(
            repo_id=model_id,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        processor_snapshot = (
            snapshot
            if processor_revision == revision
            else snapshot_download(
                repo_id=model_id,
                revision=processor_revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        )
        policy_config = PreTrainedConfig.from_pretrained(snapshot)
        policy_config.device = str(device)
        policy = SmolVLAPolicy.from_pretrained(
            snapshot,
            config=policy_config,
            strict=True,
        )
        verify_checkpoint_projection_loaded(policy, snapshot)
        policy.config.device = str(device)
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            processor_snapshot,
            preprocessor_overrides={"device_processor": {"device": str(device)}},
            postprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        return cls(
            policy,
            preprocessor,
            postprocessor,
            model_id=model_id,
            model_revision=revision,
            processor_revision=processor_revision,
            device=device,
        )

    @property
    def flow(self) -> nn.Module:
        return self.policy.model

    def preprocess_observation(self, canonical_batch: dict[str, Any]) -> dict[str, Any]:
        return self.preprocessor(dict(canonical_batch))

    def _head_modules(self) -> list[nn.Module]:
        flow = self.flow
        return [
            flow.action_in_proj,
            flow.action_out_proj,
            flow.action_time_mlp_in,
            flow.action_time_mlp_out,
        ]

    def _trainable_parameters(self, mode: FineTuningMode) -> set[int]:
        selected = _unique_parameters(self._head_modules())
        if mode == "action_expert":
            selected.update(
                id(parameter)
                for name, parameter in self.flow.vlm_with_expert.lm_expert.named_parameters()
                if "lm_head" not in name
            )
        return selected

    def configure_trainable(
        self,
        mode: FineTuningMode,
    ) -> tuple[str, ...]:
        """Select DMD2 student parameters without unfreezing the VLM backbone.

        ``action_expert`` trains every SmolVLA action-expert transformer
        parameter plus the action/time projections. ``head_only`` remains only
        to load and evaluate historical DMD2 checkpoints.
        """

        if mode not in {"action_expert", "head_only"}:
            raise ValueError("SmolVLA fine-tuning mode must be 'action_expert' or 'head_only'")
        self.policy.requires_grad_(False)
        selected = self._trainable_parameters(mode)
        for parameter in self.policy.parameters():
            parameter.requires_grad_(id(parameter) in selected)

        names = tuple(name for name, parameter in self.policy.named_parameters() if parameter.requires_grad)
        if not names:
            raise RuntimeError(f"fine-tuning mode {mode} selected no trainable parameters")
        if mode == "action_expert" and not any("vlm_with_expert.lm_expert" in name for name in names):
            raise RuntimeError("action_expert mode selected no SmolVLA expert-transformer parameters")
        if any("vlm_with_expert.vlm" in name for name in names):
            raise RuntimeError("SmolVLA VLM backbone was unexpectedly made trainable")
        self.trainable_parameter_names = names
        return names

    @torch.no_grad()
    def encode_condition(self, processed_batch: dict[str, Any]) -> SmolVLAConditionCache:
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        images, image_masks = self.policy.prepare_images(processed_batch)
        state = self.policy.prepare_state(processed_batch)
        language = processed_batch[OBS_LANGUAGE_TOKENS]
        language_mask = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix, prefix_pad, prefix_att = self.flow.embed_prefix(
            images, image_masks, language, language_mask, state=state
        )
        attention = make_att_2d_masks(prefix_pad, prefix_att)
        positions = torch.cumsum(prefix_pad, dim=1) - 1
        _, past = self.flow.vlm_with_expert.forward(
            attention_mask=attention,
            position_ids=positions,
            past_key_values=None,
            inputs_embeds=[prefix, None],
            use_cache=True,
        )
        return SmolVLAConditionCache(
            prefix_pad_masks=prefix_pad,
            past_key_values=past,
            batch_size=int(prefix_pad.shape[0]),
            prefix_length=int(prefix_pad.shape[1]),
            model_revision=self.model_revision,
            processor_revision=self.processor_revision,
            dtype=str(next(self.policy.parameters()).dtype),
            device=str(self.device),
        )

    def velocity(self, condition: SmolVLAConditionCache, x_t: Tensor, t: Tensor) -> Tensor:
        if condition.model_revision != self.model_revision:
            raise ValueError("SmolVLA condition/model revision mismatch")
        if x_t.shape != (condition.batch_size, self.chunk_size, self.internal_action_dim):
            raise ValueError(f"SmolVLA x_t must be [B,50,32], got {tuple(x_t.shape)}")
        if t.ndim == 0:
            t = t.expand(condition.batch_size)
        # The checkpoint deliberately mixes BF16 expert blocks with FP32
        # action/time projections. Keep flow coordinates at the public FP32
        # boundary and let the LeRobot module perform its internal casts. This
        # also prevents an enclosing trainer autocast from feeding BF16 into a
        # checkpoint-native FP32 projection.
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            return self.flow.denoise_step(
                prefix_pad_masks=condition.prefix_pad_masks,
                past_key_values=condition.past_key_values,
                x_t=x_t.float(),
                timestep=t.to(torch.float32),
            ).float()

    @staticmethod
    def clean_prediction(x_t: Tensor, time: Tensor, velocity: Tensor) -> Tensor:
        """Rectified-flow clean prediction ``x_hat=x_t-t*v(x_t,t)``."""

        if time.ndim == 0:
            time = time.expand(x_t.shape[0])
        if time.shape != (x_t.shape[0],) or velocity.shape != x_t.shape:
            raise ValueError("clean prediction expects x/velocity [B,H,D] and time [B]")
        return x_t.float() - time.float()[:, None, None] * velocity.float()

    def sample(
        self,
        condition: SmolVLAConditionCache,
        noise: Tensor,
        num_steps: int,
        *,
        student_time_shift_gamma: float,
        time_grid: Tensor | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> Tensor:
        """Shared differentiable sampler used by training simulation and inference."""

        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if time_grid is None:
            time_grid = shifted_time_grid(
                num_steps,
                student_time_shift_gamma,
                device=noise.device,
                dtype=torch.float32,
                descending=True,
            )
        if time_grid.shape != (num_steps + 1,) or not torch.all(time_grid[:-1] > time_grid[1:]):
            raise ValueError("time_grid must contain num_steps+1 strictly descending values")
        value = noise
        for current, target in zip(time_grid[:-1], time_grid[1:], strict=True):
            time = current.expand(condition.batch_size)
            value = value + (target - current).to(value.dtype) * self.velocity(condition, value, time)
            if step_callback is not None:
                step_callback()
        return value

    def sample_denoise_renoise(
        self,
        condition: SmolVLAConditionCache,
        noise: Tensor,
        num_steps: int,
        *,
        student_time_shift_gamma: float,
        time_grid: Tensor | None = None,
        generator: torch.Generator | None = None,
        renoise_noises: Tensor | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> Tensor:
        """DMD2 multi-step sampler with inference-matched backward simulation.

        Starting from pure Gaussian noise, each step predicts a clean sample.
        Except after the final prediction, that clean sample is independently
        re-noised at the next scheduled time. This is the denoise--renoise
        sampler in DMD2 Sections 4.4--4.5, and is intentionally distinct from
        the deterministic rectified-flow Euler sampler in :meth:`sample`.
        """

        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if noise.dtype != torch.float32:
            noise = noise.float()
        if time_grid is None:
            time_grid = shifted_time_grid(
                num_steps,
                student_time_shift_gamma,
                device=noise.device,
                dtype=torch.float32,
                descending=True,
            )
        if time_grid.shape != (num_steps + 1,) or not torch.all(
            time_grid[:-1] > time_grid[1:]
        ):
            raise ValueError("time_grid must contain num_steps+1 strictly descending values")
        return denoise_renoise_sample(
            lambda value, time: self.velocity(condition, value, time),
            noise,
            time_grid,
            generator=generator,
            renoise_noises=renoise_noises,
            step_callback=step_callback,
        )

    @torch.no_grad()
    def predict_canonical_action_chunk(
        self,
        canonical_batch: dict[str, Any],
        *,
        noise: Tensor | None = None,
        num_steps: int | None = None,
        student_time_shift_gamma: float | None = None,
    ) -> Tensor:
        processed = self.preprocess_observation(canonical_batch)
        if num_steps is not None:
            if student_time_shift_gamma is None:
                raise ValueError("a student time-grid gamma is required with a custom step count")
            condition = self.encode_condition(processed)
            if noise is None:
                noise = torch.randn(
                    condition.batch_size,
                    self.chunk_size,
                    self.internal_action_dim,
                    device=self.device,
                )
            normalized = self.sample(
                condition,
                noise,
                num_steps,
                student_time_shift_gamma=student_time_shift_gamma,
            )[..., :7]
        else:
            normalized = self.policy.predict_action_chunk(processed, noise=noise)
        return self.postprocessor(normalized)


__all__ = [
    "FineTuningMode",
    "LeRobotSmolVLAStudent",
    "SmolVLAConditionCache",
    "resolve_trainable_state_keys",
]
