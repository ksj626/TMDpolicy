from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from tmd_policy.common.density import DivergenceMode, cnf_log_density


def opd_reward(student_log_probability: Tensor, teacher_log_probability: Tensor) -> Tensor:
    if student_log_probability.shape != teacher_log_probability.shape:
        raise ValueError("student and teacher log probabilities must match")
    return -(student_log_probability - teacher_log_probability).detach()


def categorical_opd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    sampled_tokens: Tensor,
    valid_mask: Tensor,
) -> dict[str, Tensor]:
    if student_logits.shape != teacher_logits.shape or student_logits.shape[:-1] != sampled_tokens.shape:
        raise ValueError("categorical OPD logits/tokens shapes disagree")
    if valid_mask.shape != sampled_tokens.shape or valid_mask.dtype != torch.bool:
        raise ValueError("categorical OPD requires a boolean token mask")
    if torch.any(valid_mask.sum(dim=1) == 0):
        raise ValueError("each OPD trajectory needs a valid token")
    student_log_probs = torch.log_softmax(student_logits, dim=-1)
    teacher_log_probs = torch.log_softmax(teacher_logits, dim=-1)
    student_selected = student_log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    teacher_selected = teacher_log_probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    reward = opd_reward(student_selected, teacher_selected)
    per_trajectory = -(
        student_selected * reward * valid_mask
    ).sum(dim=1) / valid_mask.sum(dim=1)
    return {
        "loss": per_trajectory.mean(),
        "per_trajectory_loss": per_trajectory,
        "reward": reward,
        "student_log_probability": student_selected,
        "teacher_log_probability": teacher_selected,
    }


def continuous_flow_opd_loss(
    actions: Tensor,
    *,
    student_vector_field: Callable[[Tensor, Tensor], Tensor],
    teacher_vector_field: Callable[[Tensor, Tensor], Tensor],
    integration_steps: int,
    divergence_mode: DivergenceMode | str,
    student_probe: Tensor | None = None,
    teacher_probe: Tensor | None = None,
) -> dict[str, Tensor]:
    # OPD uses the score-function estimator: the sampled action is fixed at
    # this loss boundary while gradients still flow through the student field.
    student_log_probability = cnf_log_density(
        actions.detach(),
        student_vector_field,
        steps=integration_steps,
        mode=divergence_mode,
        probe=student_probe,
    )
    with torch.enable_grad():
        teacher_log_probability = cnf_log_density(
            actions.detach(),
            teacher_vector_field,
            steps=integration_steps,
            mode=divergence_mode,
            probe=teacher_probe,
        ).detach()
    reward = opd_reward(student_log_probability, teacher_log_probability)
    # Score-function form from VLA-OPD Eq. 7. The log-density itself retains
    # student gradients; the same value inside the reward is detached.
    per_trajectory = -(student_log_probability * reward)
    return {
        "loss": per_trajectory.mean(),
        "per_trajectory_loss": per_trajectory,
        "reward": reward,
        "student_log_probability": student_log_probability,
        "teacher_log_probability": teacher_log_probability,
    }
