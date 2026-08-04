from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ResearchConfig:
    path: Path
    value: dict[str, Any]

    @property
    def method(self) -> str:
        return str(self.value["method"])

    def validate(self) -> None:
        allowed = {
            "flow_sft", "tmd_stage1", "tmd_stage2", "tmd_plain_gaussian_ablation", "dmd2_flow", "opd_categorical",
            "continuous_flow_opd", "occupancy_discriminator", "occupancy_tmd", "data", "evaluation",
        }
        if self.method not in allowed:
            raise ValueError(f"unknown research method/operation config: {self.method}")
        for section in ("dataset", "models", "tasks", "output", "resources", "training"):
            if not isinstance(self.value.get(section), dict):
                raise TypeError(f"research config requires mapping section {section!r}")
        for name, revision in self.value["models"].get("revisions", {}).items():
            if not re.fullmatch(r"[0-9a-f]{40}", str(revision)):
                raise ValueError(f"{name} revision must be an immutable 40-character commit")
        if not re.fullmatch(r"[0-9a-f]{40}", str(self.value["dataset"].get("revision", ""))):
            raise ValueError("dataset revision must be an immutable 40-character commit")
        if self.value["training"].get("max_optimizer_steps", 0) < 1:
            raise ValueError("max_optimizer_steps must be positive")
        if self.value["resources"].get("gpus", 0) < 0:
            raise ValueError("resources.gpus cannot be negative")


def load_research_config(path: str | Path) -> ResearchConfig:
    target = Path(path).resolve()
    value = yaml.safe_load(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("research config root must be a mapping")
    config = ResearchConfig(target, value)
    config.validate()
    return config
