from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from torch import Tensor

from tmd_policy.common.capabilities import Capability


@dataclass(frozen=True)
class DryRunReport:
    method: str
    classification: str
    executable: bool
    required_capabilities: tuple[str, ...]
    missing_capabilities: tuple[str, ...]
    datasets: dict[str, str]
    checkpoints: dict[str, str]
    task_uids: tuple[str, ...]
    output_directory: str
    resource_estimate: dict[str, str]
    resolved_config: dict[str, Any]
    exact_command: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchMethod(ABC):
    name: str
    classification: str

    @abstractmethod
    def required_data_capabilities(self) -> frozenset[Capability]: ...

    @abstractmethod
    def validate_config(self) -> None: ...

    @abstractmethod
    def training_step(self, batch: Mapping[str, Any]) -> Mapping[str, Tensor]: ...

    @abstractmethod
    def sample_action_chunk(self, batch: Mapping[str, Any]) -> Tensor: ...

    @abstractmethod
    def save_method_state(self, path: str | Path) -> Path: ...

    @abstractmethod
    def load_method_state(self, path: str | Path) -> Mapping[str, Any]: ...

    @abstractmethod
    def dry_run_report(self) -> DryRunReport: ...
