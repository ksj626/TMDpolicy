from __future__ import annotations

import json
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tmd_policy.config import ConfigError, load_config
from tmd_policy.data.occupancy import DeterministicStratifiedBatchSampler
from tmd_policy.methods.discriminators import IntermediateFeatureDiscriminator
from tmd_policy.methods.dmd2_flow.program import DMD2FlowProgram
from tmd_policy.methods.flow_objectives import (
    corrupt_rectified_flow,
    executable_coordinate_mask,
    shift_time,
    stopped_dmd2_direction,
    stopped_l1_score_direction,
    surrogate_vector_loss,
)
from tmd_policy.methods.tmd.meanflow import integrate_inner_flow
from tmd_policy.methods.tmd.stage2 import TMDStage2Program
from tmd_policy.rollout import ROLLOUT_SCHEMA, ReplanRecord, RolloutEpisode, RolloutStore


def test_timestep_shift_and_rectified_corruption_match_equations() -> None:
    value = torch.tensor([0.0, 0.25, 0.5, 1.0])
    gamma = 5.0
    assert torch.allclose(shift_time(value, gamma), gamma * value / ((gamma - 1) * value + 1))
    clean = torch.zeros(4, 2, 3, requires_grad=True)
    noise = torch.ones_like(clean)
    corrupted = corrupt_rectified_flow(clean, value, noise)
    assert torch.allclose(corrupted[:, 0, 0], value)
    corrupted.sum().backward()
    assert torch.allclose(clean.grad[:, 0, 0], 1 - value)


def test_tmd_v_l1_direction_masks_padding_and_non_executable_dimensions() -> None:
    fake = torch.zeros(2, 3, 32)
    teacher = torch.zeros_like(fake)
    fake[:, :, :7] = 2
    fake[:, :, 7:] = 1_000_000
    valid = torch.tensor([[True, True, False], [True, False, False]])
    coordinates = executable_coordinate_mask(valid, 32)
    direction, metrics = stopped_l1_score_direction(
        fake, teacher, coordinates, epsilon=1.0e-6
    )
    assert torch.count_nonzero(direction[..., 7:]) == 0
    assert torch.count_nonzero(direction[0, 2]) == 0
    assert torch.allclose(metrics["valid_coordinate_count"], torch.tensor([14.0, 7.0]))
    generated = torch.zeros_like(fake, requires_grad=True)
    surrogate_vector_loss(generated, direction, coordinates).backward()
    assert torch.count_nonzero(generated.grad[..., 7:]) == 0


def test_dmd2_direction_uses_teacher_residual_mean_not_tmd_difference_l1() -> None:
    generated = torch.tensor([[[4.0, 8.0]]], requires_grad=True)
    teacher = torch.tensor([[[2.0, 4.0]]])
    fake = torch.tensor([[[3.0, 7.0]]])
    valid = torch.ones_like(generated, dtype=torch.bool)
    direction, metrics = stopped_dmd2_direction(
        fake, teacher, generated.detach(), valid, epsilon=1.0e-6
    )
    # mean(abs(generated-teacher)) = 3; fake-teacher = [1,3]
    assert torch.allclose(metrics["denominator"], torch.tensor([3.000001]), atol=1.0e-6)
    assert torch.allclose(direction, torch.tensor([[[1 / 3, 1.0]]]), atol=1.0e-5)
    surrogate_vector_loss(generated, direction, valid).backward()
    assert torch.allclose(generated.grad, direction / 2)


class _ZeroResidual(nn.Module):
    def forward(self, y_s, s, r, context):
        return torch.zeros_like(y_s)


@pytest.mark.parametrize("steps", [1, 2, 4])
def test_zero_residual_inner_flow_reproduces_smolvla_base_transition(steps: int) -> None:
    source = torch.randn(2, 5, 4)
    base = torch.randn_like(source)
    result = integrate_inner_flow(
        _ZeroResidual(),
        source,
        torch.zeros(2, 5, 3),
        num_steps=steps,
        student_time_shift_gamma=10.0,
        base_velocity=base,
    )
    assert torch.allclose(result, base, atol=1.0e-6)


def test_tmd_stage2_requires_nonzero_intended_expert_block_gradient() -> None:
    policy = nn.Module()
    policy.model = nn.Module()
    policy.model.vlm_with_expert = nn.Module()
    policy.model.vlm_with_expert.lm_expert = nn.Module()
    policy.model.vlm_with_expert.lm_expert.layers = nn.ModuleList([nn.Linear(2, 2)])
    name, parameter = next(policy.named_parameters())
    program = TMDStage2Program.__new__(TMDStage2Program)
    nn.Module.__init__(program)
    program.student = SimpleNamespace(policy=policy)
    program.intended_student_trainable_names = (name,)
    parameter.grad = torch.ones_like(parameter)
    program.validate_phase_gradients("generator")
    parameter.grad.zero_()
    with pytest.raises(RuntimeError, match="no nonzero finite gradient"):
        program.validate_phase_gradients("generator")


def test_separate_layer_heads_average_logits_and_preserve_input_gradient() -> None:
    model = IntermediateFeatureDiscriminator({2: 4, 5: 6}, hidden_dim=8)
    features = {
        2: torch.randn(3, 5, 4, requires_grad=True),
        5: torch.randn(3, 5, 6, requires_grad=True),
    }
    time = torch.tensor([0.2, 0.5, 0.8])
    valid = torch.ones(3, 5, dtype=torch.bool)
    per_layer = model.layer_logits(features, time, valid)
    aggregate = model(features, time, valid)
    assert torch.allclose(aggregate, torch.stack(list(per_layer.values())).mean(dim=0))
    torch.nn.functional.softplus(-aggregate).mean().backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in features.values())


def test_faithful_dmd_generator_never_calls_flow_sft() -> None:
    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    anchor = nn.Parameter(torch.tensor(1.0))

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = anchor

        def flow_matching_loss(self, _batch):
            raise AssertionError("faithful DMD2 must not call Flow-SFT")

    program.student = Student()
    program.dmd_config = {"gan_weight": 0.5}
    program._sample_student = types.MethodType(
        lambda self, batch, requires_grad: self.student.anchor * torch.ones(1, 50, 32), program
    )
    program._valid = types.MethodType(
        lambda self, batch, device: torch.ones(1, 50, dtype=torch.bool), program
    )
    program._distribution_matching_loss = types.MethodType(
        lambda self, batch, generated, valid: (generated.mean(), {}), program
    )
    program._generator_gan_loss = types.MethodType(
        lambda self, batch, generated, valid: generated.square().mean(), program
    )
    loss, metrics = program._generator_loss({})
    loss.backward()
    assert anchor.grad is not None
    assert "data" not in metrics


def _replan(*, environment_step: int, executed: int, truncated: bool) -> ReplanRecord:
    plan = torch.arange(350, dtype=torch.float32).reshape(50, 7)
    image = torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8)
    return ReplanRecord(
        suite="libero_spatial",
        suite_task_id=0,
        global_task_index=0,
        canonical_task_uid="libero:00:test",
        instruction="test",
        reset_seed=3,
        policy_checkpoint="checkpoint.pt",
        policy_checkpoint_sha256="a" * 64,
        policy_version="9",
        collection_round=2,
        environment_step=environment_step,
        state=torch.zeros(8),
        observations={"observation.images.image": image},
        observation_metadata={
            "observation.images.image": {
                "shape": [1, 3, 8, 8],
                "dtype": "torch.uint8",
                "layout": "BCHW",
                "encoding": "lossless torch tensor",
            }
        },
        planned_actions=plan,
        executed_prefix_length=executed,
        executed_actions=plan[:executed].clone(),
        terminated=False,
        truncated=truncated,
        success=False,
        model_revision="b" * 40,
        processor_revision="c" * 40,
        dataset_revision="d" * 40,
    )


def test_rollout_v2_round_trip_preserves_full_plans_observations_and_partial_prefix(tmp_path: Path) -> None:
    store = RolloutStore(tmp_path / "rollouts")
    store.initialize({"purpose": "unit"})
    store.append(
        RolloutEpisode(
            replans=(_replan(environment_step=0, executed=10, truncated=False), _replan(environment_step=10, executed=3, truncated=True)),
            split="train",
        )
    )
    report = store.validate()
    values = store.load_replans(store.records()[0])
    assert report["replans"] == 2 and report["steps"] == 13
    assert values[1]["planned_actions"].shape == (50, 7)
    assert torch.equal(values[1]["planned_actions"], _replan(environment_step=10, executed=3, truncated=True).planned_actions)
    assert torch.equal(values[1]["state"], torch.zeros(8))
    assert values[1]["executed_actions"].shape == (3, 7)
    assert values[1]["observations"]["observation.images.image"].dtype == torch.uint8


def test_old_rollout_schema_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "old"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema": "tmdpolicy.libero-rollout/v1"}))
    with pytest.raises(ValueError, match="cannot be reinterpreted"):
        RolloutStore(root).validate()


def test_stratified_sampler_is_source_task_paired_and_exactly_resumable() -> None:
    dataset = object.__new__(type("Dataset", (), {}))
    dataset.descriptors = [
        (task, 0, source) for task in range(3) for source in (0, 1) for _ in range(3)
    ]
    dataset.lookup = list(range(len(dataset.descriptors)))
    dataset.__class__.__len__ = lambda self: len(self.lookup)
    first = DeterministicStratifiedBatchSampler(dataset, 4, seed=11)
    batches = list(first)
    for batch in batches:
        descriptors = [dataset.descriptors[index] for index in batch]
        assert sum(source == 0 for _, _, source in descriptors) == 2
        assert sum(source == 1 for _, _, source in descriptors) == 2
    resumed = DeterministicStratifiedBatchSampler(dataset, 4, seed=11, start_batch=2)
    assert list(resumed) == batches[2:]


def test_default_baseline_configs_are_pi05_feature_based_and_have_no_data_loss() -> None:
    root = Path(__file__).resolve().parents[1]
    stage1_config = load_config(root / "configs/methods/tmd_stage1.yaml")
    assert stage1_config["tmd"]["normalization_constant_scale"] == 1.0
    assert stage1_config["tmd"]["condition_dropout_probability"] == 0.0
    for name, section in (
        ("dmd2_flow_paper.yaml", "dmd2"),
        ("tmd_stage2_paper.yaml", "stage2"),
    ):
        config = load_config(root / "configs/methods" / name)
        assert config[section]["fake_score_variant"] == "pi05_clone"
        assert config[section]["discriminator"]["variant"] == "pi05_intermediate_features"
        assert "data_weight" not in config[section]
    assert load_config(root / "configs/methods/dmd2_flow_paper.yaml")["dmd2"][
        "vsd_normalization"
    ] == "dmd2_teacher_residual_mean_abs"
    assert load_config(root / "configs/methods/dmd2_flow_paper.yaml")["dmd2"][
        "student_training_mode"
    ] == "real_data_outer_transition"
    assert load_config(root / "configs/methods/tmd_stage2_paper.yaml")["stage2"][
        "vsd_normalization"
    ] == "tmd_fake_teacher_difference_l1"
    rollout = load_config(root / "configs/rollout/student.yaml")
    assert {
        value["suite"]: set(value["task_ids"]) for value in rollout["collection"]["benchmark"]
    } == {
        "libero_spatial": set(range(10)),
        "libero_object": set(range(10)),
        "libero_goal": set(range(10)),
        "libero_10": set(range(10)),
    }


def test_configs_fail_closed_on_unknown_and_legacy_data_loss(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    original = (root / "configs/methods/dmd2_flow_paper.yaml").read_text(encoding="utf-8")
    unknown = tmp_path / "unknown.yaml"
    unknown.write_text(original + "mystery_switch: true\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown or deprecated"):
        load_config(unknown)
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text(original.replace("  gan_weight: 0.003", "  gan_weight: 0.003\n  data_weight: 1.0"))
    with pytest.raises(ConfigError, match="no SFT/data loss"):
        load_config(legacy)
    legacy_grid = tmp_path / "legacy-grid.yaml"
    legacy_grid.write_text(
        original.replace("  student_time_shift_gamma: 5.0", "  outer_time_shift_gamma: 5.0")
    )
    with pytest.raises(ConfigError, match="unknown or deprecated"):
        load_config(legacy_grid)
    tmd_original = (root / "configs/methods/tmd_stage1.yaml").read_text(encoding="utf-8")
    legacy_normalization = tmp_path / "legacy-normalization.yaml"
    legacy_normalization.write_text(
        tmd_original.replace("normalization_constant_scale: 1.0", "normalization_constant: 350.0")
    )
    with pytest.raises(ConfigError, match="unknown or deprecated"):
        load_config(legacy_normalization)


def test_pi05_evaluation_does_not_construct_smolvla_and_returns_canonical(monkeypatch) -> None:
    import tmd_policy.evaluation.policy as policy_module

    class Teacher:
        device = torch.device("cpu")
        model_id = "pi05"
        model_revision = "a" * 40
        processor_revision = "b" * 40
        policy = SimpleNamespace(config=SimpleNamespace(num_inference_steps=10))

        def preprocess_observation(self, batch):
            return batch

        def encode_condition(self, batch):
            return SimpleNamespace(batch_size=1)

        def sample(self, condition, noise, num_steps):
            assert num_steps == 10
            return noise

        def postprocess_action(self, value):
            return value[..., :7]

    monkeypatch.setattr(policy_module, "build_teacher", lambda config, device: Teacher())
    monkeypatch.setattr(
        policy_module,
        "build_student",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("SmolVLA was constructed")),
    )
    policy, identity = policy_module.load_inference_policy(
        {"policy": {"method": "pi05", "device": "cpu", "num_steps": 10}}
    )
    plan = policy.plan({"observation.state": torch.zeros(1, 8)}, noise_seed=7)
    assert plan.shape == (1, 50, 7)
    assert identity["num_steps"] == 10


def test_smolvla_official_mode_uses_checkpoint_10_steps_with_fixed_noise_parity(monkeypatch) -> None:
    import tmd_policy.evaluation.policy as policy_module

    class OfficialPolicy:
        config = SimpleNamespace(num_steps=10)

        def predict_action_chunk(self, batch, *, noise):
            return noise + 2.0

        def reset(self):
            pass

    class Student:
        device = torch.device("cpu")
        model_id = "smolvla"
        model_revision = "c" * 40
        processor_revision = "d" * 40
        policy = OfficialPolicy()

        def preprocess_observation(self, batch):
            return batch

        def postprocessor(self, value):
            return value[..., :7]

    student = Student()
    monkeypatch.setattr(policy_module, "build_student", lambda config, device: student)
    policy, identity = policy_module.load_inference_policy(
        {"policy": {"method": "smolvla", "device": "cpu", "sampler_mode": "official"}}
    )
    seed = 13
    actual = policy.plan({"observation.state": torch.zeros(1, 8)}, noise_seed=seed)
    expected_noise = torch.randn(
        1, 50, 32, generator=torch.Generator(device="cpu").manual_seed(seed)
    )
    expected = student.postprocessor(student.policy.predict_action_chunk({}, noise=expected_noise))
    assert identity["num_steps"] == 10
    assert torch.equal(actual, expected)
