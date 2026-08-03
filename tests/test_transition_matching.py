import torch
from torch import nn

from tmd_policy.models.tmd import MainBackboneOutput, TMDActionGenerator, oracle_outer_integrate
from tmd_policy.models.transition_head import InnerSourceMode, RecurrentTransitionHead


def test_outer_oracle_has_correct_sign_and_endpoint():
    action = torch.randn(4, 50, 7)
    noise = torch.randn_like(action)
    result = oracle_outer_integrate(action, noise, outer_steps=2)
    torch.testing.assert_close(result, action, atol=1e-6, rtol=1e-6)
    wrong_sign = noise + (noise - action)
    assert not torch.allclose(wrong_sign, action)


class OracleInnerHead(RecurrentTransitionHead):
    def __init__(self, target):
        super().__init__(7, 8, hidden_dim=16, num_layers=2, prediction_horizon=50)
        self.target = target

    def step(
        self,
        inner_state,
        source_noise,
        backbone_transition,
        outer_state,
        outer_time,
        inner_time,
        backbone_features,
        hidden=None,
    ):
        del inner_state, source_noise, outer_state, outer_time, inner_time
        if hidden is None:
            hidden = self.initial_hidden(backbone_features)
        return self.target - backbone_transition, hidden


def test_inner_oracle_has_correct_sign_and_endpoint():
    target = torch.randn(2, 50, 7)
    anchor = torch.randn_like(target)
    features = torch.randn(2, 50, 8)
    head = OracleInnerHead(target)
    result = head.refine(
        anchor,
        torch.zeros_like(anchor),
        torch.ones(2),
        features,
        torch.randn_like(anchor),
        inner_steps=2,
    )
    torch.testing.assert_close(result, target, atol=1e-6, rtol=1e-6)


def test_zero_initialized_head_preserves_pretrained_anchor():
    head = RecurrentTransitionHead(7, 8, hidden_dim=16, num_layers=2, prediction_horizon=50)
    anchor = torch.randn(2, 50, 7)
    result = head.refine(
        anchor,
        torch.randn_like(anchor),
        torch.ones(2),
        torch.randn(2, 50, 8),
        torch.randn_like(anchor),
        inner_steps=2,
    )
    torch.testing.assert_close(result, anchor)


def test_transition_loss_ignores_padded_positions():
    torch.manual_seed(0)
    head = RecurrentTransitionHead(7, 8, hidden_dim=16, num_layers=2, prediction_horizon=5)
    anchor = torch.randn(2, 5, 7)
    target = torch.randn_like(anchor)
    state = torch.randn_like(anchor)
    features = torch.randn(2, 5, 8)
    valid = torch.tensor([[True, True, False, False, False], [True, False, False, False, False]])
    inner_noise = torch.randn_like(target)
    first = head.matching_loss(
        anchor, target, state, torch.ones(2), features, valid, inner_noise=inner_noise
    )
    modified = target.clone()
    modified[~valid] += 1000
    second = head.matching_loss(
        anchor, modified, state, torch.ones(2), features, valid, inner_noise=inner_noise
    )
    torch.testing.assert_close(first, second)


class FixedBackbone(nn.Module):
    def forward(self, context, action_state, outer_time):
        del action_state, outer_time
        return MainBackboneOutput(context["transition"], context["features"])


def test_sampling_is_deterministic_for_fixed_noise():
    head = RecurrentTransitionHead(7, 8, hidden_dim=16, num_layers=2, prediction_horizon=50)
    generator = TMDActionGenerator(FixedBackbone(), head, outer_steps=2, inner_steps=2)
    context = {"transition": torch.randn(2, 50, 7), "features": torch.randn(2, 50, 8)}
    noise = torch.randn(2, 50, 7)
    inner_noises = torch.randn(2, 2, 50, 7)
    first = generator.sample(context, noise.clone(), inner_noises=inner_noises.clone())
    second = generator.sample(context, noise.clone(), inner_noises=inner_noises.clone())
    torch.testing.assert_close(first, second)
    assert generator.last_counts == {"main_backbone": 2, "transition_head": 4}


def test_anchored_construction_is_only_available_by_explicit_ablation_name():
    head = RecurrentTransitionHead(7, 8, hidden_dim=16, num_layers=1, prediction_horizon=5)
    anchor = torch.randn(1, 5, 7)
    result = head.refine(
        anchor,
        torch.randn_like(anchor),
        torch.ones(1),
        torch.randn(1, 5, 8),
        None,
        inner_steps=2,
        mode=InnerSourceMode.ANCHORED_TM_ABLATION,
    )
    torch.testing.assert_close(result, anchor)
