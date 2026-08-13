"""Repository-owned SmolVLA flow backend and explicit fine-tuning modes."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from tmd_policy.methods.flow_objectives import shifted_time_grid

from .compatibility import (
    validate_smolvla_instance,
    verify_checkpoint_projection_loaded,
    verify_installed_lerobot,
)

FineTuningMode = Literal["head_only", "expert_only", "lora", "full"]


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
    pooled_visual_features: Tensor
    pooled_language_features: Tensor
    pooled_state_features: Tensor
    condition_features: Tensor
    feature_identity: dict[str, Any]


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
            flow.state_proj,
            flow.action_in_proj,
            flow.action_out_proj,
            flow.action_time_mlp_in,
            flow.action_time_mlp_out,
        ]

    def configure_trainable(
        self,
        mode: FineTuningMode,
        *,
        lora_rank: int = 16,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ) -> tuple[str, ...]:
        """Select parameters by explicit module identity and validate the result."""

        if mode not in {"head_only", "expert_only", "lora", "full"}:
            raise ValueError(f"unknown SmolVLA fine-tuning mode: {mode}")
        self.policy.requires_grad_(False)
        if mode == "head_only":
            selected = _unique_parameters(self._head_modules())
            for parameter in self.policy.parameters():
                parameter.requires_grad_(id(parameter) in selected)
        elif mode == "expert_only":
            modules = [self.flow.vlm_with_expert.lm_expert, *self._head_modules()]
            selected = _unique_parameters(modules)
            for parameter in self.policy.parameters():
                parameter.requires_grad_(id(parameter) in selected)
        elif mode == "full":
            self.policy.requires_grad_(True)
        else:
            try:
                from peft import LoraConfig
            except ImportError as error:
                raise RuntimeError("lora mode requires the LeRobot training/PEFT dependencies") from error
            allowed = [self.flow.vlm_with_expert.lm_expert, *self._head_modules()]
            allowed_ids = {id(module) for root in allowed for module in root.modules()}
            targets = [
                name
                for name, module in self.policy.named_modules()
                if id(module) in allowed_ids and isinstance(module, nn.Linear)
            ]
            if not targets:
                raise RuntimeError("no exact Linear modules were found for SmolVLA LoRA")
            wrapped = self.policy.wrap_with_peft(
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=targets,
                    bias="none",
                )
            )
            self.policy = wrapped

        names = tuple(name for name, parameter in self.policy.named_parameters() if parameter.requires_grad)
        if not names:
            raise RuntimeError(f"fine-tuning mode {mode} selected no trainable parameters")
        self.trainable_parameter_names = names
        return names

    def configure_tmd_trainable(
        self, *, last_k_expert_blocks: int, train_backbone_flow_blocks: bool
    ) -> tuple[str, ...]:
        """Select the exact Stage-1/Stage-2 TMD parameter set without generic overrides."""

        layers = list(self.flow.vlm_with_expert.lm_expert.layers)
        if not 1 <= last_k_expert_blocks < len(layers):
            raise ValueError(f"last_k_expert_blocks must be in [1,{len(layers) - 1}]")
        self.policy.requires_grad_(False)
        if train_backbone_flow_blocks:
            modules = [*layers[-last_k_expert_blocks:], *self._head_modules()]
            selected = _unique_parameters(modules)
            for parameter in self.policy.parameters():
                parameter.requires_grad_(id(parameter) in selected)
        names = tuple(name for name, parameter in self.policy.named_parameters() if parameter.requires_grad)
        if train_backbone_flow_blocks and not any("lm_expert.layers" in name for name in names):
            raise RuntimeError("TMD selected no final expert-block parameters")
        self.trainable_parameter_names = names
        return names

    @property
    def condition_feature_identity(self) -> dict[str, Any]:
        hidden = int(self.flow.vlm_with_expert.config.text_config.hidden_size)
        return {
            "encoder_model_id": self.model_id,
            "encoder_model_revision": self.model_revision,
            "processor_revision": self.processor_revision,
            "feature_layer": "smolvla.embed_prefix.inputs",
            "dtype": str(next(self.policy.parameters()).dtype),
            "feature_dimension": 3 * hidden,
            "components": ["pooled_visual", "pooled_language", "pooled_state"],
            "detached": True,
        }

    def flow_matching_loss(self, canonical_batch: dict[str, Any], *, reduction: str = "mean") -> Tensor:
        processed = self.preprocess_observation(canonical_batch)
        result = self.policy(processed, reduction=reduction)
        return result[0] if isinstance(result, tuple) else result

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
        state_positions = prefix_att.bool() & prefix_pad.bool()
        state_count = state_positions.sum(dim=1)
        if torch.any(state_count == 0):
            raise RuntimeError("SmolVLA prefix contains no state-conditioned token")
        state_weight = state_positions.to(prefix.dtype).unsqueeze(-1)
        state_features = (prefix * state_weight).sum(dim=1) / state_weight.sum(dim=1).clamp_min(1)

        # LeRobot constructs prefix tokens as cameras, language, state, padding.
        # The state attention marker provides an exact boundary even when the
        # language padding mask contains false tokens.
        state_start = state_positions.float().argmax(dim=1)
        language_length = int(language.shape[1])
        positions = torch.arange(prefix.shape[1], device=prefix.device)[None]
        language_positions = (positions >= (state_start - language_length)[:, None]) & (
            positions < state_start[:, None]
        )
        language_positions &= prefix_pad.bool()
        language_weight = language_positions.to(prefix.dtype).unsqueeze(-1)
        language_features = (prefix * language_weight).sum(dim=1) / language_weight.sum(dim=1).clamp_min(1)
        visual_positions = (positions < (state_start - language_length)[:, None]) & prefix_pad.bool()
        visual_weight = visual_positions.to(prefix.dtype).unsqueeze(-1)
        visual_features = (prefix * visual_weight).sum(dim=1) / visual_weight.sum(dim=1).clamp_min(1)
        condition_features = torch.cat((visual_features, language_features, state_features), dim=-1).detach()
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
            pooled_visual_features=visual_features.detach(),
            pooled_language_features=language_features.detach(),
            pooled_state_features=state_features.detach(),
            condition_features=condition_features,
            feature_identity={
                **self.condition_feature_identity,
                "dtype": str(condition_features.dtype),
                "feature_dimension": int(condition_features.shape[-1]),
            },
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

    def velocity_with_features(
        self, condition: SmolVLAConditionCache, x_t: Tensor, t: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return raw velocity and final action-expert hidden states.

        The hook is attached by module identity to `action_out_proj`; it neither
        selects parameters nor reaches into transformer naming conventions.
        """

        captured: list[Tensor] = []

        def save_input(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            captured.append(inputs[0])

        handle = self.flow.action_out_proj.register_forward_pre_hook(save_input)
        try:
            velocity = self.velocity(condition, x_t, t)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape[:2] != x_t.shape[:2]:
            raise RuntimeError("could not capture SmolVLA action-expert features")
        return velocity, captured[0]

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
        if renoise_noises is not None and renoise_noises.shape != (
            max(0, num_steps - 1),
            *noise.shape,
        ):
            raise ValueError("renoise_noises must be [num_steps-1,B,50,32]")

        value = noise
        clean = noise
        for index, (current, target) in enumerate(
            zip(time_grid[:-1], time_grid[1:], strict=True)
        ):
            time = current.expand(condition.batch_size)
            clean = self.clean_prediction(value, time, self.velocity(condition, value, time))
            if step_callback is not None:
                step_callback()
            if index + 1 < num_steps:
                if renoise_noises is None:
                    fresh_noise = torch.randn(
                        value.shape,
                        device=value.device,
                        dtype=torch.float32,
                        generator=generator,
                    )
                else:
                    fresh_noise = renoise_noises[index].float()
                value = (1.0 - target.float()) * clean + target.float() * fresh_noise
        return clean

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
