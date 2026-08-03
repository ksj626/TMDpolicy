import torch

from tmd_policy.training.distillation import DistillationWeights, combined_distillation_loss
from tmd_policy.training.replay import ReplayRatioCorrection


def test_mismatch_weight_is_bounded_stop_gradient_and_larger_for_failures():
    logits = torch.tensor([4.0, -4.0], requires_grad=True)
    weights = DistillationWeights(0.5, 2.0, 1.0).from_expert_log_ratio(logits).values
    assert 0.5 <= weights.min() <= weights.max() <= 2.0
    assert weights[1] > weights[0]
    assert not weights.requires_grad


def test_weighted_distillation_keeps_expert_anchor():
    loss = combined_distillation_loss(
        torch.tensor([2.0, 4.0]),
        torch.tensor([1.0, 3.0]),
        torch.tensor([1.0, 3.0]),
        expert_coefficient=1.0,
        teacher_coefficient=1.0,
    )
    torch.testing.assert_close(loss, torch.tensor(5.5))


def test_replay_ratio_decomposition_and_ess():
    correction = ReplayRatioCorrection(logit_clip=2.0, weight_clip=10.0)
    first = torch.tensor([[1.0, 2.0]])
    second = torch.tensor([[0.5, 1.0]])
    combined = correction.combine(first, second)
    torch.testing.assert_close(combined, torch.tensor([[1.5, 2.0]]))
    weights = correction.weights(combined)
    assert weights.max() <= 10
    assert 1 <= correction.effective_sample_size(weights) <= weights.numel()
