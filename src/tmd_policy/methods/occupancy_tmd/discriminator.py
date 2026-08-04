from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional


class OccupancyWindowDiscriminator(nn.Module):
    """Causal state/action path model; not the DMD2 GAN discriminator."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        task_count: int,
        model_dim: int = 128,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.state_projection = nn.Linear(state_dim, model_dim)
        self.action_projection = nn.Linear(action_dim, model_dim)
        self.task_embedding = nn.Embedding(task_count, model_dim)
        block = nn.TransformerEncoderLayer(
            model_dim,
            heads,
            dim_feedforward=4 * model_dim,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(block, layers)
        self.classifier = nn.Linear(model_dim, 1)

    def forward(
        self,
        states: Tensor,
        actions: Tensor,
        task_ids: Tensor,
        valid_mask: Tensor,
    ) -> Tensor:
        if states.ndim != 3 or actions.ndim != 3 or states.shape[1] != actions.shape[1] + 1:
            raise ValueError("occupancy input must contain states [B,L+1,S] and actions [B,L,A]")
        if valid_mask.shape != actions.shape[:2] or torch.any(valid_mask.sum(dim=1) == 0):
            raise ValueError("every occupancy path needs a nonempty prefix mask")
        task = self.task_embedding(task_ids)[:, None]
        transition_tokens = self.state_projection(states[:, :-1]) + self.action_projection(actions)
        transition_tokens = transition_tokens + self.state_projection(states[:, 1:]) + task
        length = transition_tokens.shape[1]
        causal_mask = torch.triu(
            torch.ones(length, length, device=states.device, dtype=torch.bool), diagonal=1
        )
        encoded = self.transformer(
            transition_tokens,
            mask=causal_mask,
            src_key_padding_mask=~valid_mask,
        )
        return self.classifier(encoded).squeeze(-1)


def occupancy_discriminator_loss(
    expert_logits: Tensor,
    student_logits: Tensor,
    expert_mask: Tensor,
    student_mask: Tensor,
) -> Tensor:
    expert_values = expert_logits[expert_mask]
    student_values = student_logits[student_mask]
    if not len(expert_values) or not len(student_values):
        raise ValueError("balanced occupancy BCE requires both sources")
    return 0.5 * functional.softplus(-expert_values).mean() + 0.5 * functional.softplus(
        student_values
    ).mean()
