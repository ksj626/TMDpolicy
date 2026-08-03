import torch

from tmd_policy.evaluation.metrics import prefix_discriminator_report
from tmd_policy.models.discriminator import CausalPathDiscriminator, DiscriminatorVariant
from tmd_policy.training.discriminator import collate_paths


def test_prefix_logits_are_causal():
    torch.manual_seed(0)
    model = CausalPathDiscriminator(
        state_dim=8,
        action_dim=7,
        execution_horizon=10,
        num_tasks=4,
        model_dim=32,
        num_layers=2,
        num_heads=4,
        feedforward_dim=64,
        dropout=0.0,
    ).eval()
    states = torch.randn(2, 11, 8)
    actions = torch.randn(2, 10, 7)
    valid = torch.ones(2, 10, dtype=torch.bool)
    tasks = torch.tensor([0, 1])
    model.normalizer.fit(states, actions, valid, split="train")
    baseline = model(states, actions, tasks, valid)
    future_states = states.clone()
    future_actions = actions.clone()
    future_states[:, 7:] += 100
    future_actions[:, 7:] -= 100
    changed = model(future_states, future_actions, tasks, valid)
    torch.testing.assert_close(baseline[:, :6], changed[:, :6], atol=1e-5, rtol=1e-5)


def test_incremental_logits_telescope():
    logits = torch.randn(3, 10)
    increments = CausalPathDiscriminator.incremental_mismatch(logits)
    torch.testing.assert_close(increments.sum(dim=1), logits[:, -1])


def test_three_discriminator_variants_have_expected_shape():
    states = torch.randn(2, 11, 8)
    actions = torch.randn(2, 10, 7)
    tasks = torch.zeros(2, dtype=torch.long)
    valid = torch.ones(2, 10, dtype=torch.bool)
    for variant, expected in [
        (DiscriminatorVariant.POINTWISE, (2, 10)),
        (DiscriminatorVariant.FINAL, (2, 1)),
        (DiscriminatorVariant.PREFIX, (2, 10)),
    ]:
        model = CausalPathDiscriminator(
            8, 7, 10, 2, 16, 1, 4, 32, 0.0, variant
        )
        model.normalizer.fit(states, actions, valid, split="train")
        assert model(states, actions, tasks, valid).shape == expected


def test_collation_masks_short_terminal_paths_without_inventing_valid_transitions():
    batch = collate_paths(
        [
            {
                "states": torch.randn(4, 8),
                "actions": torch.randn(3, 7),
                "task_id": torch.tensor(1),
            }
        ]
    )
    assert batch["states"].shape == (1, 11, 8)
    assert batch["valid"].sum() == 3


def test_prefix_report_contains_position_task_and_failure_diagnostics():
    expert = torch.tensor([[2.0, 2.5], [1.0, 1.5]])
    student = torch.tensor([[-1.0, -2.0], [-0.5, -1.0]])
    valid = torch.ones(2, 2, dtype=torch.bool)
    report = prefix_discriminator_report(
        expert,
        student,
        valid,
        valid,
        student_success=torch.tensor([0, 1]),
        student_failure_moments=torch.tensor([[False, True], [True, False]]),
        expert_task_ids=torch.tensor([0, 1]),
        student_task_ids=torch.tensor([0, 1]),
    )
    assert len(report["prefix_by_position"]) == 2
    assert set(report["by_task"]) == {"0", "1"}
    assert "negative_increment_failure_correlation" in report
