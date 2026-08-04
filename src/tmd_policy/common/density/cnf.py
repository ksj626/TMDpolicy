from __future__ import annotations

import math
from collections.abc import Callable
from enum import StrEnum

import torch
from torch import Tensor


class DivergenceMode(StrEnum):
    EXACT = "exact"
    HUTCHINSON = "hutchinson"


def exact_divergence(velocity: Tensor, state: Tensor, *, create_graph: bool = True) -> Tensor:
    flat_velocity = velocity.reshape(velocity.shape[0], -1)
    result = torch.zeros(velocity.shape[0], device=velocity.device, dtype=velocity.dtype)
    for coordinate in range(flat_velocity.shape[1]):
        gradient = torch.autograd.grad(
            flat_velocity[:, coordinate].sum(),
            state,
            retain_graph=True,
            create_graph=create_graph,
        )[0]
        result = result + gradient.reshape(state.shape[0], -1)[:, coordinate]
    return result


def hutchinson_divergence(
    velocity: Tensor,
    state: Tensor,
    probe: Tensor,
    *,
    create_graph: bool = True,
) -> Tensor:
    if probe.shape != state.shape:
        raise ValueError("Hutchinson probe must match state shape")
    vector_jacobian = torch.autograd.grad(
        (velocity * probe).sum(), state, retain_graph=True, create_graph=create_graph
    )[0]
    return (vector_jacobian * probe).reshape(state.shape[0], -1).sum(dim=1)


def _standard_normal_log_prob(value: Tensor) -> Tensor:
    flat = value.reshape(value.shape[0], -1)
    return -0.5 * (flat.square() + math.log(2 * math.pi)).sum(dim=1)


def cnf_log_density(
    action: Tensor,
    vector_field: Callable[[Tensor, Tensor], Tensor],
    *,
    steps: int = 32,
    mode: DivergenceMode | str = DivergenceMode.EXACT,
    probe: Tensor | None = None,
) -> Tensor:
    """Invert a data-at-zero rectified flow and integrate its log Jacobian.

    This fixed-grid Euler implementation is for correctness tests and explicit
    research use; scalable solvers must be separate and cannot change its label.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    divergence_mode = DivergenceMode(mode)
    state = action
    log_jacobian = torch.zeros(action.shape[0], device=action.device, dtype=action.dtype)
    dt = 1.0 / steps
    for index in range(steps):
        state = state.requires_grad_(True)
        time = torch.full(
            (state.shape[0],), index / steps, device=state.device, dtype=state.dtype
        )
        velocity = vector_field(state, time)
        if velocity.shape != state.shape:
            raise ValueError("vector field output must match action shape")
        if divergence_mode is DivergenceMode.EXACT:
            divergence = exact_divergence(velocity, state)
        else:
            if probe is None:
                raise ValueError("Hutchinson mode requires an explicit replayable probe")
            divergence = hutchinson_divergence(velocity, state, probe)
        state = state + dt * velocity
        log_jacobian = log_jacobian + dt * divergence
    return _standard_normal_log_prob(state) + log_jacobian
