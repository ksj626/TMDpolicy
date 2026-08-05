"""Frozen PI0.5 raw rectified-flow backend for LeRobot 0.6.1."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .compatibility import (
    validate_pi05_instance,
    verify_checkpoint_projection_loaded,
    verify_installed_lerobot,
)


def _dtype(value: str | torch.dtype) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    choices = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}
    if value not in choices:
        raise ValueError(f"unsupported PI0.5 dtype {value!r}")
    return choices[value]


def _cache_tensors(cache: Any) -> list[Tensor]:
    """Discover DynamicCache tensors without depending on one storage layout."""

    found: list[Tensor] = []
    seen: set[int] = set()

    def visit(value: Any) -> None:
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, Tensor):
            found.append(value)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif hasattr(value, "__dict__"):
            for child in vars(value).values():
                visit(child)

    visit(cache)
    return found


def cache_fingerprint(cache: Any) -> tuple[tuple[int, tuple[int, ...], str, int], ...]:
    """Cheap mutation fingerprint: storage identity, shape, dtype, tensor version."""

    return tuple(
        (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype), int(tensor._version))
        for tensor in _cache_tensors(cache)
    )


@dataclass(frozen=True)
class PI05ConditionCache:
    """Observation-specific prefix state; never serialized into a dataset."""

    prefix_pad_masks: Tensor
    past_key_values: Any
    batch_size: int
    prefix_length: int
    model_id: str
    model_revision: str
    processor_revision: str
    dtype: str
    device: str
    coordinate_spec: dict[str, Any]
    fingerprint: tuple[tuple[int, tuple[int, ...], str, int], ...]


class LeRobotPI05Teacher:
    """Frozen teacher exposing dynamically evaluated raw PI0.5 velocity."""

    chunk_size = 50
    internal_action_dim = 32

    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        *,
        model_id: str,
        model_revision: str,
        processor_revision: str,
        device: str | torch.device,
        dtype: torch.dtype,
        minimum_score_time: float,
    ) -> None:
        if not 0 < minimum_score_time < 1:
            raise ValueError("minimum_score_time must be explicit and in (0,1)")
        validate_pi05_instance(policy)
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.model_id = model_id
        self.model_revision = model_revision
        self.processor_revision = processor_revision
        self.device = torch.device(device)
        self.dtype = dtype
        self.minimum_score_time = float(minimum_score_time)
        self.policy.requires_grad_(False)
        self.policy.eval()
        if int(policy.config.chunk_size) != self.chunk_size or int(policy.config.max_action_dim) != self.internal_action_dim:
            raise ValueError("loaded PI0.5 checkpoint does not implement the required [50,32] flow")

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        *,
        revision: str,
        processor_revision: str,
        device: str | torch.device = "cuda:0",
        dtype: str | torch.dtype = "bfloat16",
        minimum_score_time: float = 1e-3,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        expected_source_hashes: dict[str, str] | None = None,
    ) -> "LeRobotPI05Teacher":
        verify_installed_lerobot(expected_source_hashes=expected_source_hashes)
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05 import PI05Policy

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
        config = PreTrainedConfig.from_pretrained(snapshot)
        config.device = str(device)
        config.compile_model = False
        config.gradient_checkpointing = False
        policy = PI05Policy.from_pretrained(snapshot, config=config, strict=True).to(device)
        verify_checkpoint_projection_loaded(policy, snapshot)
        target_dtype = _dtype(dtype)
        configured_dtype = _dtype(config.dtype)
        if target_dtype != configured_dtype:
            raise ValueError(
                f"requested PI0.5 dtype {target_dtype} does not match checkpoint config dtype "
                f"{configured_dtype}; preserve the checkpoint's native mixed-precision layout"
            )
        # Do not cast the whole policy. LeRobot intentionally keeps its action
        # projections in float32 while the PaliGemma/expert backbone uses the
        # configured BF16 precision.
        policy = policy.eval()
        preprocessor, postprocessor = make_pre_post_processors(
            config,
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
            dtype=target_dtype,
            minimum_score_time=minimum_score_time,
        )

    def preprocess_observation(self, canonical_batch: dict[str, Any]) -> dict[str, Any]:
        """Use the checkpoint's official image/state/instruction processor."""

        return self.preprocessor(dict(canonical_batch))

    @torch.no_grad()
    def encode_condition(self, processed_batch: dict[str, Any]) -> PI05ConditionCache:
        from lerobot.policies.pi05.modeling_pi05 import make_att_2d_masks, prepare_attention_masks_4d
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        images, image_masks = self.policy._preprocess_images(processed_batch)
        tokens = processed_batch[OBS_LANGUAGE_TOKENS]
        masks = processed_batch[OBS_LANGUAGE_ATTENTION_MASK]
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.model.embed_prefix(
            images, image_masks, tokens, masks
        )
        prefix_attention = prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_att_masks))
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        self.policy.model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"
        _, past = self.policy.model.paligemma_with_expert.forward(
            attention_mask=prefix_attention,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        fingerprint = cache_fingerprint(past)
        if not fingerprint:
            raise RuntimeError("PI0.5 prefix forward returned an empty KV cache")
        return PI05ConditionCache(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past,
            batch_size=int(prefix_pad_masks.shape[0]),
            prefix_length=int(prefix_pad_masks.shape[1]),
            model_id=self.model_id,
            model_revision=self.model_revision,
            processor_revision=self.processor_revision,
            dtype=str(self.dtype),
            device=str(self.device),
            coordinate_spec={
                "flow": "x_t=(1-t)*action+t*noise",
                "valid_action_dimensions": 7,
                "internal_action_dimensions": 32,
                "chunk_size": 50,
            },
            fingerprint=fingerprint,
        )

    def _validate_query(self, condition: PI05ConditionCache, x_t: Tensor, t: Tensor) -> Tensor:
        if condition.model_revision != self.model_revision or condition.processor_revision != self.processor_revision:
            raise ValueError("condition cache revision does not belong to this teacher")
        if x_t.shape != (condition.batch_size, self.chunk_size, self.internal_action_dim):
            raise ValueError(f"PI0.5 x_t must be [B,50,32], got {tuple(x_t.shape)}")
        if x_t.dtype != torch.float32:
            raise ValueError("PI0.5 flow coordinates must be float32; the backbone manages its own BF16 activations")
        if t.ndim == 0:
            t = t.expand(condition.batch_size)
        if t.shape != (condition.batch_size,):
            raise ValueError("PI0.5 timestep must be scalar or [B]")
        if x_t.device != self.device or t.device != self.device:
            raise ValueError("PI0.5 query tensors must already be on the configured teacher device")
        return t.to(torch.float32)

    @torch.no_grad()
    def velocity(self, condition: PI05ConditionCache, x_t: Tensor, t: Tensor) -> Tensor:
        """Evaluate the checkpoint's one-step vector field, never action-only sampling."""

        t = self._validate_query(condition, x_t, t)
        before = cache_fingerprint(condition.past_key_values)
        if before != condition.fingerprint:
            raise RuntimeError("PI0.5 prefix cache was mutated before velocity evaluation")
        value = self.policy.model.denoise_step(
            prefix_pad_masks=condition.prefix_pad_masks,
            past_key_values=condition.past_key_values,
            x_t=x_t,
            timestep=t,
        ).detach()
        after = cache_fingerprint(condition.past_key_values)
        if after != before:
            raise RuntimeError("PI0.5 suffix evaluation mutated the retained prefix KV cache")
        if value.shape != x_t.shape:
            raise RuntimeError(f"PI0.5 raw velocity changed shape: {tuple(value.shape)} vs {tuple(x_t.shape)}")
        return value

    @property
    def action_expert_feature_dim(self) -> int:
        return int(self.policy.model.action_out_proj.in_features)

    @property
    def action_expert_layer_count(self) -> int:
        return len(self.policy.model.paligemma_with_expert.gemma_expert.model.layers)

    def intermediate_features(
        self,
        condition: PI05ConditionCache,
        noised_action: Tensor,
        time: Tensor,
        selected_layers: list[int] | tuple[int, ...],
        require_input_grad: bool,
    ) -> OrderedDict[int, Tensor]:
        """Evaluate frozen PI0.5 suffix layers while preserving action-input autograd.

        Hooks are repository-owned and attached by exact layer identity. The
        official processor and immutable prefix cache are still used; no
        installed LeRobot source is patched.
        """

        time = self._validate_query(condition, noised_action, time)
        selected = tuple(int(index) for index in selected_layers)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("selected PI0.5 feature layers must be nonempty and unique")
        layer_count = self.action_expert_layer_count
        if any(index < 0 or index >= layer_count for index in selected):
            raise ValueError(f"PI0.5 selected layers must be in [0,{layer_count - 1}]")
        before = cache_fingerprint(condition.past_key_values)
        if before != condition.fingerprint:
            raise RuntimeError("PI0.5 prefix cache was mutated before feature evaluation")

        captured: dict[int, Tensor] = {}
        handles = []

        def hook(index: int):
            def capture(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
                value = output
                if isinstance(value, (tuple, list)):
                    value = next((item for item in value if isinstance(item, Tensor)), None)
                if not isinstance(value, Tensor):
                    raise RuntimeError(f"PI0.5 layer {index} did not return tensor features")
                captured[index] = value[:, -self.chunk_size :]

            return capture

        layers = self.policy.model.paligemma_with_expert.gemma_expert.model.layers
        for index in selected:
            handles.append(layers[index].register_forward_hook(hook(index)))
        context = torch.enable_grad() if require_input_grad else torch.no_grad()
        try:
            with context:
                self.policy.model.denoise_step(
                    prefix_pad_masks=condition.prefix_pad_masks,
                    past_key_values=condition.past_key_values,
                    x_t=noised_action,
                    timestep=time,
                )
        finally:
            for handle in handles:
                handle.remove()
        after = cache_fingerprint(condition.past_key_values)
        if after != before:
            raise RuntimeError("PI0.5 feature evaluation mutated the retained prefix KV cache")
        if tuple(captured) != selected:
            raise RuntimeError(f"PI0.5 feature hooks returned layers {tuple(captured)}, expected {selected}")
        if any(value.shape[:2] != noised_action.shape[:2] for value in captured.values()):
            raise RuntimeError("PI0.5 intermediate action features changed batch/horizon shape")
        if require_input_grad and noised_action.requires_grad and not all(
            value.requires_grad for value in captured.values()
        ):
            raise RuntimeError("PI0.5 feature path detached the generated action")
        return OrderedDict((index, captured[index]) for index in selected)

    def score(self, condition: PI05ConditionCache, x_t: Tensor, t: Tensor) -> Tensor:
        """Convert rectified-flow velocity using an explicit minimum score time."""

        t = self._validate_query(condition, x_t, t)
        safe_t = t.clamp_min(self.minimum_score_time)
        velocity = self.velocity(condition, x_t, t)
        weight = safe_t[:, None, None]
        return (-(x_t + (1.0 - weight) * velocity) / weight).detach()

    @torch.no_grad()
    def sample(
        self,
        condition: PI05ConditionCache,
        noise: Tensor,
        num_steps: int,
        time_grid: Tensor | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> Tensor:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        if time_grid is None:
            time_grid = torch.linspace(1.0, 0.0, num_steps + 1, device=noise.device, dtype=torch.float32)
        if time_grid.shape != (num_steps + 1,) or not torch.all(time_grid[:-1] > time_grid[1:]):
            raise ValueError("time_grid must contain num_steps+1 strictly descending values")
        value = noise
        for current, target in zip(time_grid[:-1], time_grid[1:], strict=True):
            times = current.expand(condition.batch_size)
            value = value + (target - current).to(value.dtype) * self.velocity(condition, value, times)
            if step_callback is not None:
                step_callback()
        return value

    def postprocess_action(self, normalized_32d: Tensor) -> Tensor:
        return self.postprocessor(normalized_32d[..., :7])


__all__ = ["LeRobotPI05Teacher", "PI05ConditionCache", "cache_fingerprint"]
