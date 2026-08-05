"""Explicit fake-score variants and their fidelity labels."""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tmd_policy.backends.lerobot.pi05_teacher import (
    LeRobotPI05Teacher,
    PI05ConditionCache,
    cache_fingerprint,
)
from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent, SmolVLAConditionCache


class PI05CloneFakeScore(nn.Module):
    """PI0.5-initialized trainable suffix sharing the frozen teacher prefix."""

    classification = "paper-faithful PI0.5-initialized fake score with shared immutable prefix"

    def __init__(self, teacher: LeRobotPI05Teacher) -> None:
        super().__init__()
        self._condition_builder = teacher
        source = teacher.policy.model
        # Only suffix/action-expert computation is cloned. The frozen
        # vision/language prefix remains owned by the teacher and its KV cache
        # is passed immutably to this runner.
        self.gemma_expert = copy.deepcopy(source.paligemma_with_expert.gemma_expert)
        self.action_in_proj = copy.deepcopy(source.action_in_proj)
        self.action_out_proj = copy.deepcopy(source.action_out_proj)
        self.time_mlp_in = copy.deepcopy(source.time_mlp_in)
        self.time_mlp_out = copy.deepcopy(source.time_mlp_out)
        self.config = copy.deepcopy(source.config)
        self.model_id = teacher.model_id
        self.model_revision = teacher.model_revision
        self.processor_revision = teacher.processor_revision
        self.device = teacher.device
        self.requires_grad_(True)
        full = sum(parameter.numel() for parameter in teacher.policy.parameters())
        trainable = sum(parameter.numel() for parameter in self.parameters())
        self.parameter_report = {
            "total_teacher_parameters": full,
            "trainable_fake_score_parameters": trainable,
            "shared_frozen_parameters": full - trainable,
            "expected_parameter_memory_bytes": sum(
                parameter.numel() * parameter.element_size() for parameter in self.parameters()
            ),
            "device": str(self.device),
        }

    def verify_initialization(self) -> None:
        source = self._condition_builder.policy.model
        source_modules = {
            "gemma_expert": source.paligemma_with_expert.gemma_expert,
            "action_in_proj": source.action_in_proj,
            "action_out_proj": source.action_out_proj,
            "time_mlp_in": source.time_mlp_in,
            "time_mlp_out": source.time_mlp_out,
        }
        reference = {
            f"{prefix}.{name}": parameter
            for prefix, module in source_modules.items()
            for name, parameter in module.named_parameters()
        }
        current = dict(self.named_parameters())
        if current.keys() != reference.keys():
            raise RuntimeError("PI0.5 fake-score clone does not match the teacher suffix structure")
        for name, parameter in current.items():
            expected = reference[name]
            if parameter.shape != expected.shape or not torch.equal(
                parameter.detach(), expected.detach().to(parameter.device)
            ):
                raise RuntimeError(f"PI0.5 fake-score parameter changed before initialization check: {name}")

    def condition(self, raw_batch: dict[str, Any]) -> PI05ConditionCache:
        return self._condition_builder.encode_condition(
            self._condition_builder.preprocess_observation(raw_batch)
        )

    def _run(
        self,
        condition: PI05ConditionCache,
        x_t: Tensor,
        time: Tensor,
        selected_layers: tuple[int, ...] = (),
    ) -> tuple[Tensor, OrderedDict[int, Tensor]]:
        from lerobot.policies.pi05.modeling_pi05 import (
            clone_past_key_values,
            create_sinusoidal_pos_embedding,
            make_att_2d_masks,
            prepare_attention_masks_4d,
        )

        if x_t.shape != (condition.batch_size, 50, 32) or time.shape != (condition.batch_size,):
            raise ValueError("PI0.5 fake-score query must be actions [B,50,32] and time [B]")
        before = cache_fingerprint(condition.past_key_values)
        if before != condition.fingerprint:
            raise RuntimeError("shared PI0.5 prefix cache changed before fake-score query")
        # PI0.5 keeps its action/time projections in FP32 while the large
        # Gemma expert runs in its checkpoint-native BF16 precision.
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            time_embedding = create_sinusoidal_pos_embedding(
                time.float(),
                self.action_in_proj.out_features,
                min_period=self.config.min_period,
                max_period=self.config.max_period,
                device=time.device,
            ).float()
            suffix = self.action_in_proj(x_t.float())
            time_embedding = F.silu(
                self.time_mlp_out(F.silu(self.time_mlp_in(time_embedding)))
            )
        suffix_pad = torch.ones(x_t.shape[:2], dtype=torch.bool, device=x_t.device)
        suffix_att = torch.zeros_like(suffix_pad)
        suffix_att[:, 0] = True
        suffix_length = suffix.shape[1]
        prefix_length = condition.prefix_pad_masks.shape[1]
        prefix_mask = condition.prefix_pad_masks[:, None, :].expand(-1, suffix_length, -1)
        suffix_mask = make_att_2d_masks(suffix_pad, suffix_att)
        attention = prepare_attention_masks_4d(torch.cat((prefix_mask, suffix_mask), dim=2))
        offsets = condition.prefix_pad_masks.sum(dim=-1)[:, None]
        positions = offsets + torch.cumsum(suffix_pad, dim=1) - 1
        layers = self.gemma_expert.model.layers
        if any(index < 0 or index >= len(layers) for index in selected_layers):
            raise ValueError("fake-score selected feature layer is out of range")
        captured: dict[int, Tensor] = {}
        handles = []

        def hook(index: int):
            def capture(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
                value = output
                if isinstance(value, (tuple, list)):
                    value = next((item for item in value if isinstance(item, Tensor)), None)
                if not isinstance(value, Tensor):
                    raise RuntimeError(f"fake-score layer {index} did not expose tensor features")
                captured[index] = value[:, -50:]

            return capture

        for index in selected_layers:
            handles.append(layers[index].register_forward_hook(hook(index)))
        self.gemma_expert.model.config._attn_implementation = "eager"
        try:
            output = self.gemma_expert.model.forward(
                inputs_embeds=suffix.to(next(self.gemma_expert.parameters()).dtype),
                attention_mask=attention,
                position_ids=positions,
                past_key_values=clone_past_key_values(condition.past_key_values),
                use_cache=False,
                adarms_cond=time_embedding,
            ).last_hidden_state
        finally:
            for handle in handles:
                handle.remove()
        with torch.autocast(device_type=x_t.device.type, enabled=False):
            velocity = self.action_out_proj(output[:, -50:].float())
        if cache_fingerprint(condition.past_key_values) != before:
            raise RuntimeError("PI0.5 fake-score suffix mutated the shared prefix cache")
        return velocity, OrderedDict((index, captured[index]) for index in selected_layers)

    def forward(self, condition: PI05ConditionCache, x_t: Tensor, time: Tensor) -> Tensor:
        return self._run(condition, x_t, time)[0]

    def intermediate_features(
        self,
        condition: PI05ConditionCache,
        noised_action: Tensor,
        time: Tensor,
        selected_layers: list[int] | tuple[int, ...],
        require_input_grad: bool,
    ) -> OrderedDict[int, Tensor]:
        if not require_input_grad:
            with torch.no_grad():
                return self._run(condition, noised_action, time, tuple(selected_layers))[1]
        return self._run(condition, noised_action, time, tuple(selected_layers))[1]


class SmolVLACloneFakeScore(nn.Module):
    """Practical cross-architecture fake-score adaptation."""

    classification = "SmolVLA-clone cross-architecture adaptation; not paper-faithful"

    def __init__(self, clone: LeRobotSmolVLAStudent) -> None:
        super().__init__()
        self.clone = clone
        clone.configure_trainable("expert_only")

    def condition(self, raw_batch: dict[str, Any]) -> SmolVLAConditionCache:
        return self.clone.encode_condition(self.clone.preprocess_observation(raw_batch))

    def forward(self, condition: SmolVLAConditionCache, x_t: Tensor, time: Tensor) -> Tensor:
        return self.clone.velocity(condition, x_t, time)


__all__ = ["PI05CloneFakeScore", "SmolVLACloneFakeScore"]
