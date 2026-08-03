import pytest
import torch
from torch import nn

from tmd_policy.models.tmd import MainBackboneOutput, TMDActionGenerator, gaussian_source_like
from tmd_policy.models.transition_head import InnerSourceMode, RecurrentTransitionHead

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class FixedBackbone(nn.Module):
    def forward(self, context, action_state, outer_time):
        del action_state, outer_time
        return MainBackboneOutput(context["transition"], context["features"])


@pytest.mark.parametrize("device", DEVICES)
def test_gaussian_inner_path_and_velocity_formula(device):
    target = torch.randn(3, 5, 7, device=device)
    source = torch.randn_like(target)
    s = torch.tensor([0.0, 0.25, 1.0], device=device)
    path, velocity = RecurrentTransitionHead.gaussian_path(target, source, s)
    expected_path = (1 - s[:, None, None]) * target + s[:, None, None] * source
    torch.testing.assert_close(path, expected_path)
    torch.testing.assert_close(velocity, source - target)
    torch.testing.assert_close(path[0], target[0])
    torch.testing.assert_close(path[-1], source[-1])


@pytest.mark.parametrize("device", DEVICES)
def test_zero_residual_gaussian_integration_returns_backbone(device):
    head = RecurrentTransitionHead(7, 8, hidden_dim=16, num_layers=2, prediction_horizon=5).to(device)
    backbone = torch.randn(2, 5, 7, device=device)
    source = torch.randn_like(backbone)
    result = head.refine(
        backbone,
        torch.randn_like(backbone),
        torch.rand(2, device=device),
        torch.randn(2, 5, 8, device=device),
        source,
        inner_steps=3,
        mode=InnerSourceMode.GAUSSIAN_TM,
    )
    torch.testing.assert_close(result, backbone, atol=1e-6, rtol=1e-6)


class OracleResidualHead(RecurrentTransitionHead):
    def __init__(self, target):
        super().__init__(7, 8, hidden_dim=16, num_layers=2, prediction_horizon=5)
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


@pytest.mark.parametrize("device", DEVICES)
def test_oracle_residual_gaussian_integration_reaches_target(device):
    target = torch.randn(2, 5, 7, device=device)
    backbone = torch.randn_like(target)
    source = torch.randn_like(target)
    features = torch.randn(2, 5, 8, device=device)
    head = OracleResidualHead(target).to(device)
    result = head.refine(
        backbone,
        torch.randn_like(target),
        torch.rand(2, device=device),
        features,
        source,
        inner_steps=4,
        mode="gaussian_tm",
    )
    torch.testing.assert_close(result, target, atol=1e-6, rtol=1e-6)


def test_matching_loss_none_is_per_sample_and_respects_padding():
    torch.manual_seed(0)
    head = RecurrentTransitionHead(7, 8, hidden_dim=16, num_layers=2, prediction_horizon=5)
    backbone = torch.randn(2, 5, 7)
    target = torch.randn_like(backbone)
    source = torch.randn_like(backbone)
    outer_state = torch.randn_like(backbone)
    features = torch.randn(2, 5, 8)
    valid = torch.tensor([[True, True, False, False, False], [True, False, False, False, False]])
    first = head.matching_loss(
        backbone,
        target,
        outer_state,
        torch.rand(2),
        features,
        valid,
        inner_noise=source,
        reduction="none",
    )
    modified = target.clone()
    modified[~valid] += 1000
    second = head.matching_loss(
        backbone,
        modified,
        outer_state,
        torch.rand(2),
        features,
        valid,
        inner_noise=source,
        reduction="none",
    )
    assert first.shape == (2,)
    torch.testing.assert_close(first, second)


def test_fixed_outer_and_inner_noise_are_deterministic_and_counted():
    head = RecurrentTransitionHead(7, 8, hidden_dim=16, num_layers=2, prediction_horizon=5)
    generator = TMDActionGenerator(FixedBackbone(), head, outer_steps=2, inner_steps=3)
    context = {"transition": torch.randn(2, 5, 7), "features": torch.randn(2, 5, 8)}
    outer_noise = torch.randn(2, 5, 7)
    inner_noises = torch.randn(2, 2, 5, 7)
    first = generator.sample(context, outer_noise.clone(), inner_noises=inner_noises.clone())
    second = generator.sample(context, outer_noise.clone(), inner_noises=inner_noises.clone())
    torch.testing.assert_close(first, second)
    assert generator.last_counts == {"main_backbone": 2, "transition_head": 6}


def test_reserved_meanflow_mode_fails_explicitly():
    head = RecurrentTransitionHead(7, 8, hidden_dim=16, num_layers=1, prediction_horizon=5)
    with pytest.raises(NotImplementedError, match="reserved"):
        TMDActionGenerator(
            FixedBackbone(),
            head,
            inner_source_mode=InnerSourceMode.GAUSSIAN_TM_MEANFLOW,
        )


@pytest.mark.parametrize("device", DEVICES)
def test_primary_inner_source_is_empirically_standard_gaussian(device):
    reference = torch.empty(40_000, device=device)
    generator = torch.Generator(device=device).manual_seed(123)
    source = gaussian_source_like(reference, generator)
    assert source.shape == reference.shape
    assert source.dtype == reference.dtype
    assert source.device == reference.device
    assert abs(float(source.mean())) < 0.02
    assert abs(float(source.std(unbiased=False)) - 1.0) < 0.02


class FrozenParamBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.reference = nn.Parameter(torch.randn(1, 5, 7), requires_grad=False)
        self.feature = nn.Parameter(torch.randn(1, 5, 8), requires_grad=False)

    def forward(self, context, action_state, outer_time):
        del context, outer_time
        batch = action_state.shape[0]
        return MainBackboneOutput(
            self.reference.expand(batch, -1, -1), self.feature.expand(batch, -1, -1)
        )


def test_gradient_policy_updates_head_but_not_frozen_backbone():
    backbone = FrozenParamBackbone()
    head = RecurrentTransitionHead(7, 8, hidden_dim=16, num_layers=2, prediction_horizon=5)
    generator = TMDActionGenerator(backbone, head, outer_steps=2, inner_steps=2)
    actions = torch.randn(3, 5, 7)
    result = generator.transition_matching_loss(
        None,
        actions,
        torch.ones(3, 5, dtype=torch.bool),
        noise=torch.randn_like(actions),
        inner_noise=torch.randn_like(actions),
        outer_time=torch.full((3,), 0.5),
    )
    result["loss"].backward()
    assert head.output_projection.weight.grad is not None
    assert float(head.output_projection.weight.grad.abs().sum()) > 0
    assert backbone.reference.grad is None
    assert backbone.feature.grad is None
