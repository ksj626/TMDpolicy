"""Configuration loading for executable TMDpolicy runs.

YAML files are intentionally plain and fully resolved: there is no capability
registry and no mode that pretends to train without constructing the assets.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml

LEROBOT_VERSION = "0.6.1"
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised before any model or dataset is loaded when a run is ambiguous."""


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required configuration field: {context}.{key}")
    return mapping[key]


def _revision(value: Any, field: str) -> str:
    value = str(value)
    if not IMMUTABLE_REVISION.fullmatch(value):
        raise ConfigError(f"{field} must be an immutable 40-character Hub commit, got {value!r}")
    return value


def validate_config(config: dict[str, Any], *, expected_method: str | None = None) -> None:
    """Fail closed on asset identity and the shared LIBERO tensor contract."""

    method = str(_require(config, "method", "config"))
    if expected_method is not None and method != expected_method:
        raise ConfigError(f"command expects method={expected_method!r}, config declares {method!r}")
    backend = _require(config, "backend", "config")
    if _require(backend, "lerobot_version", "backend") != LEROBOT_VERSION:
        raise ConfigError(f"backend.lerobot_version must be exactly {LEROBOT_VERSION}")
    hashes = backend.get("expected_source_hashes")
    if hashes is not None and not isinstance(hashes, dict):
        raise ConfigError("backend.expected_source_hashes must be null or a module-name mapping")

    models = _require(config, "models", "config")
    for name in ("student", "teacher"):
        asset = _require(models, name, "models")
        if not str(_require(asset, "id", f"models.{name}")).strip():
            raise ConfigError(f"models.{name}.id must be nonempty")
        _revision(_require(asset, "revision", f"models.{name}"), f"models.{name}.revision")
        _revision(
            _require(asset, "processor_revision", f"models.{name}"),
            f"models.{name}.processor_revision",
        )

    dataset = _require(config, "dataset", "config")
    if _require(dataset, "id", "dataset") != "lerobot/libero":
        raise ConfigError("the canonical expert dataset must be lerobot/libero")
    _revision(_require(dataset, "revision", "dataset"), "dataset.revision")
    fractions = [float(dataset.get("validation_fraction", 0.1)), float(dataset.get("test_fraction", 0.1))]
    if min(fractions) <= 0 or sum(fractions) >= 1:
        raise ConfigError("validation/test fractions must be positive and sum to less than one")

    horizons = _require(config, "horizons", "config")
    if int(_require(horizons, "prediction", "horizons")) != 50:
        raise ConfigError("LIBERO prediction horizon is fixed at 50")
    execution = int(_require(horizons, "execution", "horizons"))
    if not 1 <= execution <= 50:
        raise ConfigError("horizons.execution must be in [1, 50]")

    training = config.get("training")
    if training is not None:
        if int(training.get("batch_size", 0)) < 1 or int(training.get("max_steps", 0)) < 1:
            raise ConfigError("training.batch_size and training.max_steps must be positive")
        if training.get("mixed_precision", "bf16") not in {"no", "fp16", "bf16"}:
            raise ConfigError("training.mixed_precision must be no, fp16, or bf16")
        if int(training.get("gradient_accumulation", 0)) < 1:
            raise ConfigError("training.gradient_accumulation must be positive")

    if "capabilities" in json.dumps(config):
        raise ConfigError("capability declarations were removed; configure concrete assets instead")


def load_config(path: str | Path, *, expected_method: str | None = None) -> dict[str, Any]:
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    config = copy.deepcopy(value)
    config["_config_path"] = str(path.resolve())
    validate_config(config, expected_method=expected_method)
    return config


def save_resolved_config(config: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


__all__ = [
    "ConfigError",
    "IMMUTABLE_REVISION",
    "LEROBOT_VERSION",
    "PROJECT_ROOT",
    "load_config",
    "project_path",
    "save_resolved_config",
    "validate_config",
]
