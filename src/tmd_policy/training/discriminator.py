from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from tmd_policy.models.discriminator import CausalPathDiscriminator, DiscriminatorVariant


def _task_position_balanced_bce(
    logits: Tensor,
    mask: Tensor,
    task_ids: Tensor,
    label: float,
) -> Tensor:
    elementwise = F.binary_cross_entropy_with_logits(
        logits, torch.full_like(logits, label), reduction="none"
    )
    groups: list[Tensor] = []
    for task in torch.unique(task_ids):
        task_rows = task_ids == task
        for position in range(logits.shape[1]):
            selected = task_rows & mask[:, position]
            if selected.any():
                groups.append(elementwise[selected, position].mean())
    if not groups:
        raise ValueError("balanced discriminator loss received no valid task-position group")
    return torch.stack(groups).mean()


def task_balanced_indices(task_ids: Tensor, *, generator: torch.Generator | None = None) -> Tensor:
    """Subsample the same number of paths from every represented task."""

    unique, counts = torch.unique(task_ids, return_counts=True)
    if not len(unique):
        raise ValueError("cannot balance an empty task list")
    count = int(counts.min())
    selected: list[Tensor] = []
    for task in unique:
        indices = torch.nonzero(task_ids == task, as_tuple=False).squeeze(1)
        order = torch.randperm(len(indices), device=indices.device, generator=generator)
        selected.append(indices[order[:count]])
    return torch.cat(selected)


@contextmanager
def frozen_module(module: nn.Module) -> Iterator[nn.Module]:
    """Temporarily freeze a scoring model during a student update."""

    training = module.training
    policies = [parameter.requires_grad for parameter in module.parameters()]
    module.eval().requires_grad_(False)
    try:
        yield module
    finally:
        module.train(training)
        for parameter, requires_grad in zip(module.parameters(), policies, strict=True):
            parameter.requires_grad_(requires_grad)


def discriminator_loss(
    model: CausalPathDiscriminator,
    expert: dict[str, Tensor],
    student: dict[str, Tensor],
) -> tuple[Tensor, dict[str, Tensor]]:
    expert_logits = model(expert["states"], expert["actions"], expert["task_ids"], expert["valid"])
    student_logits = model(student["states"], student["actions"], student["task_ids"], student["valid"])
    if model.variant == DiscriminatorVariant.FINAL:
        expert_mask = torch.ones_like(expert_logits, dtype=torch.bool)
        student_mask = torch.ones_like(student_logits, dtype=torch.bool)
    else:
        expert_mask = expert["valid"]
        student_mask = student["valid"]
    positive = _task_position_balanced_bce(
        expert_logits, expert_mask, expert["task_ids"], label=1.0
    )
    negative = _task_position_balanced_bce(
        student_logits, student_mask, student["task_ids"], label=0.0
    )
    loss = 0.5 * (positive + negative)
    return loss, {
        "expert_logits": expert_logits.detach(),
        "student_logits": student_logits.detach(),
        "expert_bce": positive.detach(),
        "student_bce": negative.detach(),
    }


def train_discriminator_step(
    model: CausalPathDiscriminator,
    optimizer: torch.optim.Optimizer,
    expert: dict[str, Tensor],
    student: dict[str, Tensor],
    *,
    max_grad_norm: float = 5.0,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss, details = discriminator_loss(model, expert, student)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "expert_bce": float(details["expert_bce"]),
        "student_bce": float(details["student_bce"]),
    }


def collate_paths(samples: list[dict[str, Tensor]], execution_horizon: int = 10) -> dict[str, Tensor]:
    if not samples:
        raise ValueError("cannot collate an empty path batch")
    state_dim = samples[0]["states"].shape[-1]
    action_dim = samples[0]["actions"].shape[-1]
    states = torch.zeros(len(samples), execution_horizon + 1, state_dim, dtype=torch.float32)
    actions = torch.zeros(len(samples), execution_horizon, action_dim, dtype=torch.float32)
    valid = torch.zeros(len(samples), execution_horizon, dtype=torch.bool)
    tasks = torch.empty(len(samples), dtype=torch.long)
    for index, sample in enumerate(samples):
        length = min(execution_horizon, sample["actions"].shape[0])
        states[index, : length + 1] = sample["states"][: length + 1]
        actions[index, :length] = sample["actions"][:length]
        sample_valid = sample.get("valid", torch.ones(length, dtype=torch.bool))[:length]
        valid[index, :length] = sample_valid
        tasks[index] = int(sample["task_id"])
    return {"states": states, "actions": actions, "valid": valid, "task_ids": tasks}
