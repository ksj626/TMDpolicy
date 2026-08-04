"""Narrow interfaces consumed by distillation algorithms.

Algorithms depend on these protocols rather than arbitrary LeRobot objects.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from torch import Tensor


class CanonicalBatch(TypedDict, total=False):
    """Canonical LIBERO batch before a checkpoint's official processor.

    Images are float tensors ``[B,3,H,W]`` in ``[0,1]``; state is ``[B,8]``;
    action is ``[B,50,7]``; ``action_is_pad`` is boolean ``[B,50]``; and
    ``task`` is a list of strings.
    """

    observation_state: Tensor
    action: Tensor
    action_is_pad: Tensor
    episode_index: Tensor
    frame_index: Tensor
    task_index: Tensor
    task: list[str]


class FlowCondition(Protocol):
    batch_size: int
    device: str
    dtype: str


class FlowPolicy(Protocol):
    chunk_size: int
    internal_action_dim: int

    def preprocess_observation(self, canonical_batch: dict[str, Any]) -> dict[str, Any]: ...

    def encode_condition(self, processed_batch: dict[str, Any]) -> FlowCondition: ...

    def velocity(self, condition: FlowCondition, x_t: Tensor, t: Tensor) -> Tensor: ...

    def sample(
        self,
        condition: FlowCondition,
        noise: Tensor,
        num_steps: int,
        time_grid: Tensor | None = None,
    ) -> Tensor: ...


__all__ = ["CanonicalBatch", "FlowCondition", "FlowPolicy"]
