"""Repository-owned SmolVLA flow backend and explicit fine-tuning modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .compatibility import (
    validate_smolvla_instance,
    verify_checkpoint_projection_loaded,
    verify_installed_lerobot,
)

FineTuningMode = Literal["head_only", "expert_only", "lora", "full"]


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
        policy = SmolVLAPolicy.from_pretrained(snapshot, strict=True).to(device)
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
        return self.flow.denoise_step(
            prefix_pad_masks=condition.prefix_pad_masks,
            past_key_values=condition.past_key_values,
            x_t=x_t,
            timestep=t.to(torch.float32),
        )

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
        time_grid: Tensor | None = None,
    ) -> Tensor:
        """Shared differentiable sampler used by training simulation and inference."""

        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if time_grid is None:
            time_grid = torch.linspace(1.0, 0.0, num_steps + 1, device=noise.device, dtype=torch.float32)
        if time_grid.shape != (num_steps + 1,) or not torch.all(time_grid[:-1] > time_grid[1:]):
            raise ValueError("time_grid must contain num_steps+1 strictly descending values")
        value = noise
        for current, target in zip(time_grid[:-1], time_grid[1:], strict=True):
            time = current.expand(condition.batch_size)
            value = value + (target - current).to(value.dtype) * self.velocity(condition, value, time)
        return value

    @torch.no_grad()
    def predict_canonical_action_chunk(
        self,
        canonical_batch: dict[str, Any],
        *,
        noise: Tensor | None = None,
        num_steps: int | None = None,
    ) -> Tensor:
        processed = self.preprocess_observation(canonical_batch)
        if num_steps is not None:
            condition = self.encode_condition(processed)
            if noise is None:
                noise = torch.randn(
                    condition.batch_size,
                    self.chunk_size,
                    self.internal_action_dim,
                    device=self.device,
                )
            normalized = self.sample(condition, noise, num_steps)[..., :7]
        else:
            normalized = self.policy.predict_action_chunk(processed, noise=noise)
        return self.postprocessor(normalized)


__all__ = ["FineTuningMode", "LeRobotSmolVLAStudent", "SmolVLAConditionCache"]
