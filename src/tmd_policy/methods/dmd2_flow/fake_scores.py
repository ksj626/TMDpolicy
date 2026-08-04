"""Explicit fake-score variants and their fidelity labels."""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from tmd_policy.backends.lerobot.pi05_teacher import LeRobotPI05Teacher, PI05ConditionCache
from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent, SmolVLAConditionCache


class PI05CloneFakeScore(nn.Module):
    """Trainable PI0.5 action-expert clone: closest DMD2 variant, very expensive."""

    classification = "closest-to-original PI0.5 fake-score clone; memory intensive"

    def __init__(self, clone: LeRobotPI05Teacher) -> None:
        super().__init__()
        self.policy = clone.policy
        self.preprocessor = clone.preprocessor
        self._cache_builder = clone
        self.policy.requires_grad_(False)
        flow = self.policy.model
        modules = [
            flow.paligemma_with_expert.gemma_expert,
            flow.action_in_proj,
            flow.action_out_proj,
            flow.time_mlp_in,
            flow.time_mlp_out,
        ]
        selected = {id(parameter) for module in modules for parameter in module.parameters()}
        for parameter in self.policy.parameters():
            parameter.requires_grad_(id(parameter) in selected)

    def condition(self, raw_batch: dict[str, Any]) -> PI05ConditionCache:
        return self._cache_builder.encode_condition(self.preprocessor(dict(raw_batch)))

    def forward(self, condition: PI05ConditionCache, x_t: Tensor, time: Tensor) -> Tensor:
        return self.policy.model.denoise_step(
            prefix_pad_masks=condition.prefix_pad_masks,
            past_key_values=condition.past_key_values,
            x_t=x_t,
            timestep=time,
        )


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
