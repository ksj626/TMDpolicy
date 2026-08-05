"""Differentiable action normalization and PI0.5/SmolVLA coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


def safe_device_transfer(value: Tensor, device: str | torch.device) -> Tensor:
    """Transfer tensors without relying on CUDA peer-to-peer copies.

    The supported multi-GPU workstation intermittently corrupts direct peer
    copies under the current PyTorch/CUDA stack. Two explicit copy operations
    preserve autograd while routing cross-GPU values through host memory.
    """

    target = torch.device(device)
    if value.device == target:
        return value
    if value.device.type == "cuda" and target.type == "cuda":
        return value.to("cpu", non_blocking=False).to(target, non_blocking=False)
    return value.to(target)


def _mode_name(value: Any) -> str:
    return str(getattr(value, "value", value)).upper()


class ActionNormalizer(nn.Module):
    """Torch-only affine action transform matching LeRobot 0.6.1.

    ``normalize`` maps canonical environment actions to checkpoint coordinates;
    ``unnormalize`` performs the inverse. Statistics are registered buffers, so
    moving the bridge never traverses NumPy/CPU in the training path.
    """

    def __init__(self, mode: str, stats: dict[str, Tensor], *, eps: float = 1e-8) -> None:
        super().__init__()
        self.mode = _mode_name(mode)
        self.eps = float(eps)
        supported = {"IDENTITY", "MEAN_STD", "MIN_MAX", "QUANTILES"}
        if self.mode not in supported:
            raise ValueError(f"unsupported action normalization {self.mode}; supported: {sorted(supported)}")
        required = {
            "IDENTITY": (),
            "MEAN_STD": ("mean", "std"),
            "MIN_MAX": ("min", "max"),
            "QUANTILES": ("q01", "q99"),
        }[self.mode]
        missing = set(required).difference(stats)
        if missing:
            raise ValueError(f"{self.mode} requires action statistics {sorted(missing)}")
        for name in required:
            value = torch.as_tensor(stats[name]).detach().clone().to(torch.float32)
            if value.ndim != 1:
                raise ValueError(f"action statistic {name} must be one-dimensional")
            self.register_buffer(name, value, persistent=True)

    @property
    def action_dim(self) -> int:
        if self.mode == "IDENTITY":
            raise RuntimeError("IDENTITY normalizer has no intrinsic dimension")
        first = next(self.buffers())
        return int(first.numel())

    @classmethod
    def from_pipeline(cls, pipeline: Any) -> "ActionNormalizer":
        """Extract the loaded action mode/stats from an official processor."""

        candidates = [step for step in pipeline.steps if hasattr(step, "norm_map") and hasattr(step, "_tensor_stats")]
        if len(candidates) != 1:
            raise ValueError(f"expected one normalization step in processor, found {len(candidates)}")
        step = candidates[0]
        action_modes = [value for key, value in step.norm_map.items() if _mode_name(key) == "ACTION"]
        if len(action_modes) != 1 or "action" not in step._tensor_stats:
            raise ValueError("processor does not expose one ACTION normalization mapping and action statistics")
        return cls(action_modes[0], step._tensor_stats["action"], eps=float(step.eps))

    def _stats_like(self, tensor: Tensor, *names: str) -> list[Tensor]:
        return [getattr(self, name).to(device=tensor.device, dtype=tensor.dtype) for name in names]

    def normalize(self, canonical: Tensor) -> Tensor:
        if self.mode == "IDENTITY":
            return canonical
        if self.mode == "MEAN_STD":
            mean, std = self._stats_like(canonical, "mean", "std")
            return (canonical - mean) / (std + self.eps)
        low_name, high_name = ("min", "max") if self.mode == "MIN_MAX" else ("q01", "q99")
        low, high = self._stats_like(canonical, low_name, high_name)
        span = torch.where(high == low, torch.full_like(high, self.eps), high - low)
        return 2.0 * (canonical - low) / span - 1.0

    def unnormalize(self, normalized: Tensor) -> Tensor:
        if self.mode == "IDENTITY":
            return normalized
        if self.mode == "MEAN_STD":
            mean, std = self._stats_like(normalized, "mean", "std")
            return normalized * std + mean
        low_name, high_name = ("min", "max") if self.mode == "MIN_MAX" else ("q01", "q99")
        low, high = self._stats_like(normalized, low_name, high_name)
        span = torch.where(high == low, torch.full_like(high, self.eps), high - low)
        return (normalized + 1.0) * span / 2.0 + low


@dataclass(frozen=True)
class CoordinateResult:
    values: Tensor
    timestep_valid: Tensor
    dimension_valid: Tensor


class ActionCoordinateBridge(nn.Module):
    """Implements ``z_T = N_T(N_S^{-1}(z_S))`` and 7→32 padding."""

    def __init__(
        self,
        student: ActionNormalizer,
        teacher: ActionNormalizer,
        *,
        environment_dim: int = 7,
        teacher_internal_dim: int = 32,
    ) -> None:
        super().__init__()
        if environment_dim != 7 or teacher_internal_dim != 32:
            raise ValueError("the audited LIBERO/PI0.5 coordinate contract is 7→32")
        if student.action_dim != environment_dim or teacher.action_dim != environment_dim:
            raise ValueError("loaded student/teacher action statistics must both have dimension 7")
        self.student = student
        self.teacher = teacher
        self.environment_dim = environment_dim
        self.teacher_internal_dim = teacher_internal_dim
        self.register_buffer(
            "teacher_dimension_valid",
            torch.arange(teacher_internal_dim) < environment_dim,
            persistent=False,
        )

    @classmethod
    def from_processors(
        cls,
        student_preprocessor: Any,
        teacher_preprocessor: Any,
        *,
        student_config: Any | None = None,
        teacher_config: Any | None = None,
    ) -> "ActionCoordinateBridge":
        cls.validate_action_semantics(student_config, teacher_config)
        return cls(
            ActionNormalizer.from_pipeline(student_preprocessor),
            ActionNormalizer.from_pipeline(teacher_preprocessor),
        )

    @staticmethod
    def validate_action_semantics(student_config: Any | None, teacher_config: Any | None) -> None:
        flags = {}
        for label, config in (("student", student_config), ("teacher", teacher_config)):
            if config is None:
                continue
            for name in ("adapt_to_pi_aloha", "use_delta_joint_actions_aloha"):
                flags[f"{label}.{name}"] = bool(getattr(config, name, False))
        enabled = [name for name, value in flags.items() if value]
        if enabled:
            raise ValueError(
                "relative/ALOHA action semantics cannot be reconciled with canonical LIBERO Cartesian actions: "
                + ", ".join(enabled)
            )

    def student_to_canonical(self, student_values: Tensor) -> Tensor:
        if student_values.shape[-1] not in {self.environment_dim, self.teacher_internal_dim}:
            raise ValueError("student action must end in 7 valid or 32 padded dimensions")
        return self.student.unnormalize(student_values[..., : self.environment_dim])

    def canonical_to_student(self, canonical: Tensor) -> Tensor:
        if canonical.shape[-1] != self.environment_dim:
            raise ValueError("canonical LIBERO action must have 7 dimensions")
        return self.student.normalize(canonical)

    def canonical_to_teacher(self, canonical: Tensor, valid_mask: Tensor | None = None) -> CoordinateResult:
        if canonical.shape[-1] != self.environment_dim:
            raise ValueError("canonical LIBERO action must have 7 dimensions")
        normalized = self.teacher.normalize(canonical)
        padded = torch.zeros(*normalized.shape[:-1], self.teacher_internal_dim, device=normalized.device, dtype=normalized.dtype)
        padded[..., : self.environment_dim] = normalized
        if valid_mask is None:
            valid_mask = torch.ones(normalized.shape[:-1], device=normalized.device, dtype=torch.bool)
        if valid_mask.shape != normalized.shape[:-1] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean and match action batch/time axes")
        padded = padded * valid_mask.unsqueeze(-1).to(padded.dtype)
        dimensions = self.teacher_dimension_valid.to(device=padded.device)
        return CoordinateResult(padded, valid_mask, dimensions)

    def student_to_teacher(self, student_values: Tensor, valid_mask: Tensor | None = None) -> CoordinateResult:
        return self.canonical_to_teacher(self.student_to_canonical(student_values), valid_mask)

    def teacher_to_canonical(self, teacher_values: Tensor) -> Tensor:
        if teacher_values.shape[-1] not in {self.environment_dim, self.teacher_internal_dim}:
            raise ValueError("teacher action must end in 7 valid or 32 padded dimensions")
        return self.teacher.unnormalize(teacher_values[..., : self.environment_dim])

    def teacher_to_student(self, teacher_values: Tensor) -> Tensor:
        return self.canonical_to_student(self.teacher_to_canonical(teacher_values))


__all__ = [
    "ActionCoordinateBridge",
    "ActionNormalizer",
    "CoordinateResult",
    "safe_device_transfer",
]
