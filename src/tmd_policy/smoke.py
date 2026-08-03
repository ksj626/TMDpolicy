from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from tmd_policy.evaluation.metrics import discriminator_report
from tmd_policy.models.discriminator import CausalPathDiscriminator
from tmd_policy.models.tmd import MainBackboneOutput, TMDActionGenerator, oracle_outer_integrate
from tmd_policy.models.transition_head import RecurrentTransitionHead
from tmd_policy.training.discriminator import discriminator_loss
from tmd_policy.training.distillation import DistillationWeights
from tmd_policy.training.replay import ReplayRatioCorrection


class _ToyBackbone(nn.Module):
    def __init__(self, action_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.feature = nn.Linear(action_dim, feature_dim)

    def forward(self, context: dict[str, Tensor], action_state: Tensor, outer_time: Tensor) -> MainBackboneOutput:
        del outer_time
        features = self.feature(context["target"])
        return MainBackboneOutput(transition=context["anchor"], features=features)


def run_synthetic_smoke(output_dir: str | Path, seed: int = 7) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    batch, horizon, action_dim, feature_dim = 8, 50, 7, 32
    actions = torch.randn(batch, horizon, action_dim, device=device).clamp(-1, 1)
    noise = torch.randn_like(actions)
    target = noise - actions
    anchor = target + 0.4 * torch.tanh(actions)
    backbone = _ToyBackbone(action_dim, feature_dim)
    head = RecurrentTransitionHead(action_dim, feature_dim, 64, 2, horizon)
    generator = TMDActionGenerator(backbone, head, outer_steps=2, inner_steps=2)
    context = {"target": actions, "anchor": anchor}
    optimizer = torch.optim.AdamW(head.parameters(), lr=3e-3)
    losses: list[float] = []
    started = time.perf_counter()
    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        result = generator.transition_matching_loss(
            context,
            actions,
            torch.ones(batch, horizon, dtype=torch.bool),
            noise=noise,
            outer_time=torch.full((batch,), 0.5),
        )
        result["loss"].backward()
        optimizer.step()
        losses.append(float(result["loss"].detach()))
    training_time = time.perf_counter() - started
    oracle_error = float((oracle_outer_integrate(actions, noise, 2) - actions).abs().max())
    fixed_inner = torch.randn(2, *noise.shape)
    fixed_one = generator.sample(context, noise.clone(), inner_noises=fixed_inner.clone()).detach()
    fixed_two = generator.sample(context, noise.clone(), inner_noises=fixed_inner.clone()).detach()

    path_batch, execution_horizon, state_dim = 32, 10, 8
    expert_states = torch.randn(path_batch, execution_horizon + 1, state_dim) * 0.2
    expert_actions = torch.randn(path_batch, execution_horizon, action_dim) * 0.2
    student_states = expert_states + torch.linspace(0, 0.8, execution_horizon + 1)[None, :, None]
    student_actions = expert_actions + 0.35
    valid = torch.ones(path_batch, execution_horizon, dtype=torch.bool)
    tasks = torch.arange(path_batch) % 4
    expert_batch = {"states": expert_states, "actions": expert_actions, "valid": valid, "task_ids": tasks}
    student_batch = {
        "states": student_states,
        "actions": student_actions,
        "valid": valid,
        "task_ids": tasks,
    }
    discriminator = CausalPathDiscriminator(
        state_dim=state_dim,
        action_dim=action_dim,
        execution_horizon=execution_horizon,
        num_tasks=4,
        model_dim=32,
        num_layers=2,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
    )
    discriminator.normalizer.fit(
        torch.cat((expert_states, student_states)),
        torch.cat((expert_actions, student_actions)),
        torch.cat((valid, valid)),
        split="train",
    )
    disc_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=2e-3)
    for _ in range(30):
        disc_optimizer.zero_grad(set_to_none=True)
        disc_loss, _ = discriminator_loss(discriminator, expert_batch, student_batch)
        disc_loss.backward()
        disc_optimizer.step()
    discriminator.eval()
    with torch.no_grad():
        expert_logits = discriminator(states=expert_states, actions=expert_actions, task_ids=tasks, valid_mask=valid)
        student_logits = discriminator(states=student_states, actions=student_actions, task_ids=tasks, valid_mask=valid)
        disc_metrics = discriminator_report(expert_logits.numpy(), student_logits.numpy())
        final_student = discriminator.final_prefix(student_logits, valid)
        weights = DistillationWeights().from_expert_log_ratio(final_student).values
    replay = ReplayRatioCorrection()
    replay_weights = replay.weights(replay.combine(student_logits, -0.25 * student_logits))
    report: dict[str, object] = {
        "mode": "synthetic_tiny_data_smoke",
        "executed": True,
        "tmd": {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_decreased": losses[-1] < losses[0],
            "oracle_endpoint_max_error": oracle_error,
            "deterministic_fixed_noise": bool(torch.equal(fixed_one, fixed_two)),
            "main_backbone_evaluations": generator.last_counts["main_backbone"],
            "recurrent_head_evaluations": generator.last_counts["transition_head"],
            "training_wall_time_s": training_time,
        },
        "discriminator": disc_metrics,
        "distillation": {
            "minimum_weight": float(weights.min()),
            "maximum_weight": float(weights.max()),
            "stop_gradient": not weights.requires_grad,
        },
        "replay": {
            "effective_sample_size": float(replay.effective_sample_size(replay_weights)),
            "maximum_weight": float(replay_weights.max()),
        },
    }
    report_path = output / "synthetic_smoke_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
