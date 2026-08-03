from __future__ import annotations

import inspect
import subprocess
from pathlib import Path
from typing import Any


class LeRobotCompatibilityError(RuntimeError):
    pass


def _require_parameters(callable_object: Any, required: tuple[str, ...], label: str) -> None:
    available = tuple(inspect.signature(callable_object).parameters)
    missing = [name for name in required if name not in available]
    if missing:
        raise LeRobotCompatibilityError(
            f"LeRobot {label} signature is incompatible: missing {missing}; available={available}. "
            "Use the commit pinned by checkpoints.lerobot_commit."
        )


def _lerobot_checkout() -> Path:
    import lerobot

    source = Path(lerobot.__file__).resolve()
    for parent in source.parents:
        if (parent / ".git").exists():
            return parent
    raise LeRobotCompatibilityError(f"cannot locate a git checkout above {source}")


def verify_lerobot_api(expected_commit: str, policy: Any | None = None) -> dict[str, Any]:
    """Fail early if the exact SmolVLA internals consumed by the adapter changed."""

    from lerobot.policies.smolvla.modeling_smolvla import (
        SmolVLAPolicy,
        VLAFlowMatching,
        make_att_2d_masks,
    )

    checkout = _lerobot_checkout()
    commit = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_commit:
        raise LeRobotCompatibilityError(
            f"LeRobot commit mismatch: expected {expected_commit}, found {commit} at {checkout}"
        )
    _require_parameters(SmolVLAPolicy.prepare_images, ("self", "batch"), "prepare_images")
    _require_parameters(SmolVLAPolicy.prepare_state, ("self", "batch"), "prepare_state")
    _require_parameters(SmolVLAPolicy.prepare_action, ("self", "batch"), "prepare_action")
    _require_parameters(
        VLAFlowMatching.embed_prefix,
        ("self", "images", "img_masks", "lang_tokens", "lang_masks", "state"),
        "embed_prefix",
    )
    _require_parameters(
        VLAFlowMatching.embed_suffix,
        ("self", "noisy_actions", "timestep"),
        "embed_suffix",
    )
    _require_parameters(make_att_2d_masks, ("pad_masks", "att_masks"), "make_att_2d_masks")
    if policy is not None:
        required_policy = ("model", "config", "prepare_images", "prepare_state", "prepare_action")
        missing_policy = [name for name in required_policy if not hasattr(policy, name)]
        if missing_policy:
            raise LeRobotCompatibilityError(f"SmolVLA policy instance lacks {missing_policy}")
        flow = policy.model
        required_flow = (
            "vlm_with_expert",
            "embed_prefix",
            "embed_suffix",
            "action_out_proj",
            "sample_noise",
        )
        missing_flow = [name for name in required_flow if not hasattr(flow, name)]
        if missing_flow:
            raise LeRobotCompatibilityError(f"SmolVLA flow instance lacks {missing_flow}")
        _require_parameters(
            flow.vlm_with_expert.forward,
            (
                "attention_mask",
                "position_ids",
                "past_key_values",
                "inputs_embeds",
                "use_cache",
                "fill_kv_cache",
            ),
            "vlm_with_expert.forward",
        )
    return {
        "compatible": True,
        "expected_commit": expected_commit,
        "installed_commit": commit,
        "checkout": str(checkout),
        "runtime_policy_checked": policy is not None,
    }


__all__ = ["LeRobotCompatibilityError", "verify_lerobot_api"]
