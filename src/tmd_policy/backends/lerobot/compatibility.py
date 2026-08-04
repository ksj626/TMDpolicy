"""Runtime contract for the pinned PyPI LeRobot package."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch

EXPECTED_VERSION = "0.6.1"


class LeRobotCompatibilityError(RuntimeError):
    """An installed LeRobot API used by TMDpolicy does not match the pin."""


_MODULES = {
    "pi05_modeling": "lerobot.policies.pi05.modeling_pi05",
    "smolvla_modeling": "lerobot.policies.smolvla.modeling_smolvla",
    "normalization_processor": "lerobot.processor.normalize_processor",
    "policy_processor_factory": "lerobot.policies.factory",
}

_SIGNATURES = {
    "PI05Pytorch.embed_prefix": ("self", "images", "img_masks", "tokens", "masks"),
    "PI05Pytorch.denoise_step": (
        "self",
        "prefix_pad_masks",
        "past_key_values",
        "x_t",
        "timestep",
    ),
    "PI05Policy._preprocess_images": ("self", "batch"),
    "SmolVLAPolicy.forward": ("self", "batch", "noise", "time", "reduction"),
    "SmolVLAPolicy.prepare_images": ("self", "batch"),
    "SmolVLAPolicy.prepare_state": ("self", "batch"),
    "VLAFlowMatching.denoise_step": (
        "self",
        "prefix_pad_masks",
        "past_key_values",
        "x_t",
        "timestep",
    ),
    "make_pre_post_processors": (
        "policy_cfg",
        "pretrained_path",
        "pretrained_revision",
        "kwargs",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parameters(callable_value: Any) -> tuple[str, ...]:
    return tuple(inspect.signature(callable_value).parameters)


def verify_installed_lerobot(
    expected_version: str = EXPECTED_VERSION,
    expected_source_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate every private-ish API consumed by repository-owned adapters.

    Source hashes are always recorded. Passing expected hashes turns them into
    strict reproducibility checks without requiring a local Git checkout.
    """

    try:
        installed = version("lerobot")
    except PackageNotFoundError as error:
        raise LeRobotCompatibilityError(
            "LeRobot is not installed; run scripts/setup/create_environment.sh"
        ) from error
    if installed != expected_version:
        raise LeRobotCompatibilityError(
            f"TMDpolicy requires lerobot=={expected_version}, found {installed}; recreate the environment"
        )

    loaded = {name: importlib.import_module(module_name) for name, module_name in _MODULES.items()}
    pi = loaded["pi05_modeling"]
    smol = loaded["smolvla_modeling"]
    objects = {
        "PI05Pytorch.embed_prefix": pi.PI05Pytorch.embed_prefix,
        "PI05Pytorch.denoise_step": pi.PI05Pytorch.denoise_step,
        "PI05Policy._preprocess_images": pi.PI05Policy._preprocess_images,
        "SmolVLAPolicy.forward": smol.SmolVLAPolicy.forward,
        "SmolVLAPolicy.prepare_images": smol.SmolVLAPolicy.prepare_images,
        "SmolVLAPolicy.prepare_state": smol.SmolVLAPolicy.prepare_state,
        "VLAFlowMatching.denoise_step": smol.VLAFlowMatching.denoise_step,
        "make_pre_post_processors": loaded["policy_processor_factory"].make_pre_post_processors,
    }
    signatures: dict[str, list[str]] = {}
    for name, expected in _SIGNATURES.items():
        actual = _parameters(objects[name])
        signatures[name] = list(actual)
        if actual != expected:
            raise LeRobotCompatibilityError(
                f"incompatible LeRobot API {name}: expected parameters {expected}, found {actual}"
            )

    required = {
        "PI05Policy": (pi.PI05Policy, ("from_pretrained", "predict_action_chunk")),
        "PI05Pytorch": (
            pi.PI05Pytorch,
            ("embed_prefix", "embed_suffix", "denoise_step", "sample_actions"),
        ),
        "SmolVLAPolicy": (
            smol.SmolVLAPolicy,
            ("from_pretrained", "forward", "prepare_action", "predict_action_chunk"),
        ),
        "VLAFlowMatching": (
            smol.VLAFlowMatching,
            ("embed_prefix", "embed_suffix", "denoise_step", "sample_actions"),
        ),
    }
    for object_name, (object_value, attributes) in required.items():
        missing = [attribute for attribute in attributes if not hasattr(object_value, attribute)]
        if missing:
            raise LeRobotCompatibilityError(f"{object_name} is missing required attributes: {missing}")

    sources: dict[str, dict[str, str]] = {}
    expected_source_hashes = expected_source_hashes or {}
    unknown = set(expected_source_hashes).difference(_MODULES)
    if unknown:
        raise LeRobotCompatibilityError(f"unknown expected source-hash keys: {sorted(unknown)}")
    for name, module in loaded.items():
        path = Path(inspect.getfile(module)).resolve()
        digest = _sha256(path)
        sources[name] = {"module": module.__name__, "path": str(path), "sha256": digest}
        expected = expected_source_hashes.get(name)
        if expected is not None and digest != expected:
            raise LeRobotCompatibilityError(
                f"source hash mismatch for {name}: expected {expected}, found {digest} at {path}"
            )
    return {"lerobot_version": installed, "signatures": signatures, "sources": sources}


def validate_pi05_instance(policy: Any) -> None:
    """Validate instance-owned modules after checkpoint construction."""

    required = {
        "policy.model": getattr(policy, "model", None),
        "policy.model.paligemma_with_expert": getattr(getattr(policy, "model", None), "paligemma_with_expert", None),
        "policy.model.action_out_proj": getattr(getattr(policy, "model", None), "action_out_proj", None),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise LeRobotCompatibilityError(f"loaded PI0.5 checkpoint lacks required objects: {missing}")


def validate_smolvla_instance(policy: Any) -> None:
    model = getattr(policy, "model", None)
    required = {
        "policy.model": model,
        "policy.model.vlm_with_expert": getattr(model, "vlm_with_expert", None),
        "policy.model.vlm_with_expert.lm_expert": getattr(
            getattr(model, "vlm_with_expert", None), "lm_expert", None
        ),
        "policy.model.action_out_proj": getattr(model, "action_out_proj", None),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise LeRobotCompatibilityError(f"loaded SmolVLA checkpoint lacks required objects: {missing}")


def verify_checkpoint_projection_loaded(policy: Any, snapshot: str | Path) -> None:
    """Detect PI0.5's upstream catch-and-return-uninitialized failure mode."""

    path = Path(snapshot) / "model.safetensors"
    if not path.is_file():
        raise LeRobotCompatibilityError(f"immutable checkpoint is missing {path}")
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            expected = handle.get_tensor("model.action_out_proj.weight")
    except (KeyError, OSError) as error:
        raise LeRobotCompatibilityError(
            f"cannot validate model.action_out_proj.weight in {path}"
        ) from error
    state = policy.state_dict()
    key = "model.action_out_proj.weight"
    if key not in state:
        raise LeRobotCompatibilityError(f"loaded policy state lacks {key}")
    actual = state[key].detach().cpu()
    if actual.shape != expected.shape or not torch.equal(actual, expected.to(actual.dtype)):
        raise LeRobotCompatibilityError(
            "checkpoint action projection does not match model.safetensors; upstream loading may have "
            "returned an uninitialized policy"
        )


__all__ = [
    "EXPECTED_VERSION",
    "LeRobotCompatibilityError",
    "validate_pi05_instance",
    "validate_smolvla_instance",
    "verify_checkpoint_projection_loaded",
    "verify_installed_lerobot",
]
