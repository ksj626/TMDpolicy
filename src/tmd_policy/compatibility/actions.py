from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor


class Processor(Protocol):
    def __call__(self, value: Any) -> Any: ...


@dataclass(frozen=True)
class CanonicalActionSpace:
    dimension: int = 7
    low: float = -1.0
    high: float = 1.0

    def project(self, actions: Tensor) -> Tensor:
        if actions.shape[-1] != self.dimension:
            raise ValueError(f"expected canonical action dim {self.dimension}, got {actions.shape}")
        return actions.clamp(self.low, self.high)

    def validate(self, actions: Tensor, *, tolerate: float = 1e-4) -> None:
        if actions.shape[-1] != self.dimension:
            raise ValueError(f"expected canonical action dim {self.dimension}, got {actions.shape}")
        if not torch.isfinite(actions).all():
            raise ValueError("canonical actions contain non-finite values")
        if actions.numel() and (actions.min() < self.low - tolerate or actions.max() > self.high + tolerate):
            raise ValueError(f"canonical actions outside [{self.low}, {self.high}]")


class PolicyActionBridge:
    """Converts only through official policy processors.

    `to_internal` passes canonical actions through the policy preprocessor;
    `to_canonical` passes policy outputs through its postprocessor. No loss is
    permitted between two models' internal normalized spaces.
    """

    def __init__(
        self,
        preprocessor: Processor,
        postprocessor: Processor,
        action_space: CanonicalActionSpace | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.action_space = action_space or CanonicalActionSpace()

    def to_internal(self, canonical: Tensor, context: dict[str, Any] | None = None) -> Tensor:
        self.action_space.validate(canonical)
        batch = dict(context or {})
        batch["action"] = canonical
        processed = self.preprocessor(batch)
        return processed["action"]

    def to_canonical(
        self, internal: Tensor, *, project_to_environment: bool = True, validate_bounds: bool = True
    ) -> Tensor:
        canonical = self.postprocessor(internal)
        if isinstance(canonical, dict):
            canonical = canonical["action"]
        canonical = torch.as_tensor(canonical)
        if project_to_environment:
            canonical = self.action_space.project(canonical)
        if validate_bounds:
            self.action_space.validate(canonical, tolerate=5e-3)
        return canonical

    def round_trip_error(self, canonical: Tensor, context: dict[str, Any] | None = None) -> float:
        restored = self.to_canonical(self.to_internal(canonical, context), project_to_environment=False)
        return float((restored.to(canonical) - canonical).abs().max().item())


@dataclass(frozen=True)
class StateCompatibilityAdapter:
    canonical_dim: int = 8
    student_dim: int = 8

    def for_student(self, state: Tensor) -> Tensor:
        if state.shape[-1] != self.canonical_dim:
            raise ValueError(f"canonical LIBERO state must be {self.canonical_dim}D, got {state.shape}")
        return state[..., : self.student_dim]

    def for_teacher(self, state: Tensor) -> Tensor:
        if state.shape[-1] != self.canonical_dim:
            raise ValueError(f"canonical LIBERO state must be {self.canonical_dim}D, got {state.shape}")
        return state
