"""Shared deterministic LIBERO evaluation and rollout protocol."""

from __future__ import annotations

from typing import Any, Mapping


LIBERO_SUITES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

# Cross-method benchmark horizons, fixed at LeRobot's 280-step spatial protocol
# and recorded in every resolved run configuration.
LIBERO_SUITE_MAX_EPISODE_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

LIBERO_FIXED_INIT_STATE_COUNT = 50


def validate_suite_max_episode_steps(value: Any, *, context: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a suite-to-positive-step mapping")
    normalized = {str(suite): int(steps) for suite, steps in value.items()}
    if set(normalized) != set(LIBERO_SUITES) or min(normalized.values(), default=0) < 1:
        raise ValueError(f"{context} must cover all four LIBERO suites with positive values")
    return normalized


def init_state_index_for_trial(trial: int) -> int:
    if int(trial) < 0:
        raise ValueError("LIBERO trial/init-state indices must be nonnegative")
    return int(trial) % LIBERO_FIXED_INIT_STATE_COUNT


def set_fixed_init_state_index(env: Any, index: int) -> int:
    """Set the next fixed state on a one-or-more-subenv vector environment."""

    normalized = init_state_index_for_trial(index)
    if not hasattr(env, "set_attr"):
        raise TypeError("LIBERO vector environment does not expose set_attr")
    env.set_attr("init_state_id", normalized)
    return normalized


__all__ = [
    "LIBERO_FIXED_INIT_STATE_COUNT",
    "LIBERO_SUITES",
    "LIBERO_SUITE_MAX_EPISODE_STEPS",
    "init_state_index_for_trial",
    "set_fixed_init_state_index",
    "validate_suite_max_episode_steps",
]
