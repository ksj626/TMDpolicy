from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from tmd_policy.models.tmd import MainBackboneOutput, TMDActionGenerator
from tmd_policy.models.transition_head import RecurrentTransitionHead


class DiagnosticBackbone(nn.Module):
    """Deterministic reference used only to audit loss optimization."""

    def __init__(self, action_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.feature_projection = nn.Linear(action_dim, feature_dim)
        self.requires_grad_(False)

    def forward(
        self, context: dict[str, Tensor], action_state: Tensor, outer_time: Tensor
    ) -> MainBackboneOutput:
        del action_state, outer_time
        return MainBackboneOutput(
            transition=context["reference"],
            features=self.feature_projection(context["actions"]),
        )


def overfit_action_chunk(
    actions: Tensor,
    *,
    steps: int = 80,
    seed: int = 7,
    learning_rate: float = 3e-3,
) -> tuple[dict[str, Any], list[float]]:
    """Overfit Gaussian TM on one canonical action chunk for a regression diagnostic."""

    if actions.shape != (1, 50, 7):
        raise ValueError(f"expected one canonical [1,50,7] chunk, got {actions.shape}")
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning_rate must be positive")
    torch.manual_seed(seed)
    batch_actions = actions.float().repeat(4, 1, 1)
    outer_noise = torch.randn_like(batch_actions)
    target = outer_noise - batch_actions
    reference = target + 0.35 * torch.tanh(batch_actions)
    backbone = DiagnosticBackbone(7, 32).to(actions.device)
    head = RecurrentTransitionHead(7, 32, hidden_dim=64, num_layers=2, prediction_horizon=50).to(
        actions.device
    )
    generator = TMDActionGenerator(backbone, head, outer_steps=2, inner_steps=2)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    context = {"actions": batch_actions, "reference": reference}
    valid = torch.ones(4, 50, dtype=torch.bool, device=actions.device)
    inner_noise = torch.randn_like(batch_actions)
    losses: list[float] = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        result = generator.transition_matching_loss(
            context,
            batch_actions,
            valid,
            noise=outer_noise,
            inner_noise=inner_noise,
            outer_time=torch.full((4,), 0.5, device=actions.device),
        )
        result["loss"].backward()
        optimizer.step()
        losses.append(float(result["loss"].detach()))
    return (
        {
            "inner_source_mode": generator.inner_source_mode.value,
            "steps": steps,
            "seed": seed,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_decreased": losses[-1] < losses[0],
            "reduction_factor": losses[0] / max(losses[-1], 1e-12),
        },
        losses,
    )


def run_npz_chunk_overfit(
    payload: str | Path,
    output_dir: str | Path,
    *,
    steps: int = 80,
    seed: int = 7,
    device: str = "cpu",
) -> dict[str, Any]:
    source = Path(payload)
    with np.load(source, allow_pickle=False) as data:
        actions = torch.from_numpy(data["plan_actions"])[None].to(device)
    report, losses = overfit_action_chunk(actions, steps=steps, seed=seed)
    report.update(
        {
            "data_label": "real expert action chunk; diagnostic backbone features",
            "source_payload": str(source.resolve()),
            "action_shape": list(actions.shape),
            "action_min": float(actions.min()),
            "action_max": float(actions.max()),
        }
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("step", "transition_matching_loss"))
        writer.writerows(enumerate(losses))
    return report


__all__ = ["DiagnosticBackbone", "overfit_action_chunk", "run_npz_chunk_overfit"]
