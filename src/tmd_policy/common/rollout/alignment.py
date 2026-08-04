from __future__ import annotations

import numpy as np


def validate_rollout_alignment(states: np.ndarray, actions: np.ndarray, valid: np.ndarray) -> None:
    if states.ndim != 2 or actions.ndim != 2 or states.shape[0] != actions.shape[0] + 1:
        raise ValueError("executed rollout must align as L actions and L+1 states")
    if valid.dtype != np.bool_ or valid.shape != (actions.shape[0],) or not valid.all():
        raise ValueError("stored executed prefixes contain real transitions only")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("rollout contains nonfinite values")
