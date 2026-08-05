from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from tmd_policy.backends.action_coordinates import (
    ActionCoordinateBridge,
    ActionNormalizer,
    safe_device_transfer,
)
from tmd_policy.backends.lerobot.compatibility import verify_installed_lerobot
from tmd_policy.backends.lerobot.pi05_teacher import LeRobotPI05Teacher, PI05ConditionCache, cache_fingerprint


def test_installed_lerobot_contract_and_source_hashes() -> None:
    report = verify_installed_lerobot()
    assert report["lerobot_version"] == "0.6.1"
    assert set(report["sources"]) == {
        "pi05_modeling",
        "smolvla_modeling",
        "normalization_processor",
        "policy_processor_factory",
    }
    assert all(len(value["sha256"]) == 64 for value in report["sources"].values())


@pytest.mark.parametrize(
    ("mode", "stats"),
    [
        ("MEAN_STD", {"mean": torch.arange(7.0), "std": torch.arange(1.0, 8.0)}),
        ("MIN_MAX", {"min": -torch.arange(1.0, 8.0), "max": torch.arange(1.0, 8.0)}),
        ("QUANTILES", {"q01": -torch.arange(1.0, 8.0), "q99": torch.arange(1.0, 8.0)}),
    ],
)
def test_normalization_round_trip(mode: str, stats: dict[str, torch.Tensor]) -> None:
    normalizer = ActionNormalizer(mode, stats)
    value = torch.linspace(-0.7, 0.7, 14).reshape(2, 7)
    assert torch.allclose(normalizer.unnormalize(normalizer.normalize(value)), value, atol=1e-6)


@pytest.mark.parametrize("mode", ["MEAN_STD", "MIN_MAX", "QUANTILES"])
def test_action_transform_matches_official_lerobot_processor(mode: str) -> None:
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep

    stats = {
        "action": {
            "mean": torch.linspace(-0.2, 0.2, 7),
            "std": torch.linspace(0.5, 1.1, 7),
            "min": torch.linspace(-2.0, -1.0, 7),
            "max": torch.linspace(1.0, 2.0, 7),
            "q01": torch.linspace(-1.5, -0.7, 7),
            "q99": torch.linspace(0.8, 1.7, 7),
        }
    }
    features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))}
    norm_map = {FeatureType.ACTION: NormalizationMode(mode)}
    official_normalize = NormalizerProcessorStep(features, norm_map, stats)
    official_unnormalize = UnnormalizerProcessorStep(features, norm_map, stats)
    ours = ActionNormalizer(mode, stats["action"])
    canonical = torch.randn(3, 50, 7)
    normalized = ours.normalize(canonical)
    assert torch.allclose(normalized, official_normalize._normalize_action(canonical, inverse=False), atol=1e-6)
    assert torch.allclose(
        ours.unnormalize(normalized),
        official_unnormalize._normalize_action(normalized, inverse=True),
        atol=1e-6,
    )


def test_coordinate_bridge_padding_mask_and_gradient() -> None:
    student = ActionNormalizer("MEAN_STD", {"mean": torch.zeros(7), "std": torch.arange(1.0, 8.0)})
    teacher = ActionNormalizer("MIN_MAX", {"min": -torch.ones(7), "max": 2 * torch.ones(7)})
    bridge = ActionCoordinateBridge(student, teacher)
    value = torch.randn(2, 50, 7, requires_grad=True)
    valid = torch.ones(2, 50, dtype=torch.bool)
    valid[:, -3:] = False
    result = bridge.student_to_teacher(value, valid)
    assert result.values.shape == (2, 50, 32)
    assert result.dimension_valid.tolist() == [True] * 7 + [False] * 25
    assert torch.count_nonzero(result.values[:, -3:]) == 0
    assert torch.count_nonzero(result.values[..., 7:]) == 0
    result.values[..., :7].square().mean().backward()
    assert value.grad is not None and torch.count_nonzero(value.grad[:, :-3]) > 0


def test_safe_device_transfer_preserves_values_and_autograd() -> None:
    value = torch.randn(3, 4, requires_grad=True)
    transferred = safe_device_transfer(value, "cpu")
    assert transferred is value
    transferred.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


class _FakeCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.paligemma_with_expert = SimpleNamespace()
        self.action_out_proj = nn.Identity()

    def denoise_step(self, *, prefix_pad_masks, past_key_values, x_t, timestep):
        return x_t * 0 + self.weight


class _FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _FakeCore()
        self.config = SimpleNamespace(chunk_size=50, max_action_dim=32)


def _fake_teacher() -> tuple[LeRobotPI05Teacher, PI05ConditionCache]:
    policy = _FakePolicy()
    teacher = LeRobotPI05Teacher(
        policy,
        object(),
        object(),
        model_id="teacher",
        model_revision="a" * 40,
        processor_revision="b" * 40,
        device="cpu",
        dtype=torch.float32,
        minimum_score_time=0.1,
    )
    past = SimpleNamespace(key_cache=[torch.ones(1, 1, 2, 3)], value_cache=[torch.ones(1, 1, 2, 3)])
    fingerprint = cache_fingerprint(past)
    cache = PI05ConditionCache(
        prefix_pad_masks=torch.ones(1, 2, dtype=torch.bool),
        past_key_values=past,
        batch_size=1,
        prefix_length=2,
        model_id="teacher",
        model_revision="a" * 40,
        processor_revision="b" * 40,
        dtype="torch.float32",
        device="cpu",
        coordinate_spec={},
        fingerprint=fingerprint,
    )
    return teacher, cache


def test_teacher_velocity_shape_cache_immutability_and_no_parameter_gradient() -> None:
    teacher, cache = _fake_teacher()
    x = torch.randn(1, 50, 32, requires_grad=True)
    velocity = teacher.velocity(cache, x, torch.tensor([0.5]))
    assert velocity.shape == x.shape
    assert cache_fingerprint(cache.past_key_values) == cache.fingerprint
    assert not velocity.requires_grad
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in teacher.policy.parameters())


def test_score_conversion_formula_and_explicit_clamp() -> None:
    teacher, cache = _fake_teacher()
    x = torch.randn(1, 50, 32)
    time = torch.tensor([0.05])
    velocity = torch.full_like(x, 3.0)

    def fixed(self, condition, query, query_time):
        return velocity

    teacher.velocity = MethodType(fixed, teacher)
    score = teacher.score(cache, x, time)
    safe = torch.tensor(0.1)
    expected = -(x + (1 - safe) * velocity) / safe
    assert torch.allclose(score, expected)


def test_relative_action_settings_fail_closed() -> None:
    with pytest.raises(ValueError, match="cannot be reconciled"):
        ActionCoordinateBridge.validate_action_semantics(
            SimpleNamespace(adapt_to_pi_aloha=False, use_delta_joint_actions_aloha=True),
            SimpleNamespace(),
        )
