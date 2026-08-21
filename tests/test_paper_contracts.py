from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tmd_policy.cli import build_parser
from tmd_policy.config import ConfigError, load_config
from tmd_policy.evaluation.policy import _load_student_checkpoint_state
from tmd_policy.methods.dmd2_flow.discriminator import IntermediateFeatureDiscriminator
from tmd_policy.methods.dmd2_flow.program import DMD2FlowProgram, _stable_first_order_surrogate
from tmd_policy.methods.flow_objectives import (
    corrupt_rectified_flow,
    executable_coordinate_mask,
    shift_time,
    stopped_dmd2_direction,
    surrogate_vector_loss,
)
from tmd_policy.rollout import ROLLOUT_SCHEMA, ReplanRecord, RolloutEpisode, RolloutStore


def test_timestep_shift_and_rectified_corruption_match_equations() -> None:
    value = torch.tensor([0.0, 0.25, 0.5, 1.0])
    gamma = 5.0
    assert torch.allclose(shift_time(value, gamma), gamma * value / ((gamma - 1) * value + 1))
    clean = torch.zeros(4, 2, 3, requires_grad=True)
    corrupted = corrupt_rectified_flow(clean, value, torch.ones_like(clean))
    assert torch.allclose(corrupted[:, 0, 0], value)
    corrupted.sum().backward()
    assert torch.allclose(clean.grad[:, 0, 0], 1 - value)


def test_first_order_surrogate_recovers_safe_scaled_gradient() -> None:
    class FiniteOnlyWhenScaled(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value.sum()

        @staticmethod
        def backward(ctx, output_gradient):
            if float(output_gradient.abs()) > 0.01:
                return torch.full((3,), float("inf"))
            return output_gradient.expand(3) * 4.0

    value = torch.ones(3, requires_grad=True)
    surrogate, scale, gradient = _stable_first_order_surrogate(
        FiniteOnlyWhenScaled.apply(value), value, scales=(1.0, 2.0**-8)
    )
    surrogate.backward()
    assert scale == 2.0**-8
    assert torch.equal(gradient, torch.full((3,), 4.0))
    assert torch.equal(value.grad, gradient)


def test_dmd2_direction_uses_teacher_residual_mean() -> None:
    generated = torch.tensor([[[4.0, 8.0]]], requires_grad=True)
    teacher = torch.tensor([[[2.0, 4.0]]])
    fake = torch.tensor([[[3.0, 7.0]]])
    valid = torch.ones_like(generated, dtype=torch.bool)
    direction, metrics = stopped_dmd2_direction(
        fake, teacher, generated.detach(), valid, epsilon=1.0e-6
    )
    assert torch.allclose(metrics["denominator"], torch.tensor([3.000001]), atol=1.0e-6)
    assert torch.allclose(direction, torch.tensor([[[1 / 3, 1.0]]]), atol=1.0e-5)
    surrogate_vector_loss(generated, direction, valid).backward()
    assert torch.allclose(generated.grad, direction / 2)


def test_executable_mask_excludes_padding_and_latent_dimensions() -> None:
    valid = torch.tensor([[True, False]])
    coordinates = executable_coordinate_mask(valid, 32)
    assert coordinates.shape == (1, 2, 32)
    assert coordinates[0, 0, :7].all()
    assert not coordinates[0, 0, 7:].any()
    assert not coordinates[0, 1].any()


def test_feature_heads_are_fp32_and_preserve_input_gradient() -> None:
    model = IntermediateFeatureDiscriminator({2: 4, 5: 6}, hidden_dim=8)
    features = {
        2: torch.randn(3, 5, 4, requires_grad=True),
        5: torch.randn(3, 5, 6, requires_grad=True),
    }
    time = torch.tensor([0.2, 0.5, 0.8])
    valid = torch.ones(3, 5, dtype=torch.bool)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        per_layer = model.layer_logits(features, time, valid)
        aggregate = model(features, time, valid)
    assert aggregate.dtype == torch.float32
    assert torch.allclose(aggregate, torch.stack(list(per_layer.values())).mean(dim=0))
    torch.nn.functional.softplus(-aggregate).mean().backward()
    assert all(value.grad is not None and value.grad.abs().sum() > 0 for value in features.values())


def test_model_parallel_to_never_migrates_fake_score() -> None:
    class Tracked(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.moves = []

        def to(self, *args, **kwargs):
            self.moves.append((args, kwargs))
            return super().to(*args, **kwargs)

    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    program.student, program.bridge, program.fake_score, program.discriminator = Tracked(), Tracked(), Tracked(), Tracked()
    program._fake_score_device = torch.device("cpu")
    program._discriminator_device = torch.device("cpu")
    program.to("cpu")
    assert len(program.student.moves) == len(program.bridge.moves) == 1
    assert program.fake_score.moves == [] and program.discriminator.moves == []


def _replan(*, environment_step: int, executed: int, truncated: bool) -> ReplanRecord:
    plan = torch.arange(350, dtype=torch.float32).reshape(50, 7)
    image = torch.randint(0, 256, (1, 3, 8, 8), dtype=torch.uint8)
    return ReplanRecord(
        suite="libero_spatial", suite_task_id=0, global_task_index=0,
        canonical_task_uid="libero:00:test", instruction="test", reset_seed=3,
        init_state_index=7, policy_checkpoint="checkpoint.pt", policy_checkpoint_sha256="a" * 64,
        policy_version="9", collection_round=2, environment_step=environment_step,
        state=torch.zeros(8), observations={"observation.images.image": image},
        observation_metadata={"observation.images.image": {"shape": [1, 3, 8, 8], "dtype": "torch.uint8", "layout": "BCHW", "encoding": "lossless torch tensor"}},
        planned_actions=plan, executed_prefix_length=executed, executed_actions=plan[:executed].clone(),
        terminated=False, truncated=truncated, success=False, model_revision="b" * 40,
        processor_revision="c" * 40, dataset_revision="d" * 40,
    )


def test_replan_distinguishes_suite_local_and_global_task_ids() -> None:
    values = dict(_replan(environment_step=0, executed=10, truncated=False).__dict__)
    values.update(global_task_index=30, canonical_task_uid="libero:30:test")
    record = ReplanRecord(**values)
    assert record.suite_task_id == 0 and record.global_task_index == 30
    values.update(canonical_task_uid="libero:29:test")
    with pytest.raises(ValueError, match="dataset-global"):
        ReplanRecord(**values)


def test_rollout_round_trip_preserves_full_plan_and_prefix(tmp_path: Path) -> None:
    store = RolloutStore(tmp_path / "rollouts")
    store.initialize({"purpose": "unit"})
    store.append(RolloutEpisode(replans=(_replan(environment_step=0, executed=3, truncated=True),), split="train"))
    assert ROLLOUT_SCHEMA in (tmp_path / "rollouts/manifest.json").read_text()
    report = store.validate()
    values = store.load_replans(store.records()[0])[0]
    assert report["replans"] == 1 and report["steps"] == 3
    assert values["planned_actions"].shape == (50, 7)
    assert values["executed_actions"].shape == (3, 7)


def test_old_rollout_schema_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "old"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema": "tmdpolicy.libero-rollout/v1"}))
    with pytest.raises(ValueError, match="cannot be reinterpreted"):
        RolloutStore(root).validate()


def test_canonical_dmd2_config_matches_retained_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/methods/dmd2_flow.yaml")
    dmd2 = config["dmd2"]
    assert dmd2["fake_score_variant"] == "pi05_clone"
    assert dmd2["discriminator"]["variant"] == "pi05_intermediate_features"
    assert dmd2["discriminator"]["feature_source"] == "fake_score_features"
    assert dmd2["vsd_normalization"] == "dmd2_teacher_residual_mean_abs"
    assert dmd2["student_fine_tuning"] == "action_expert"
    assert config["training"]["batch_size"] * config["training"]["gradient_accumulation"] == 32


@pytest.mark.parametrize("removed", ["flow_sft", "tmd_stage1", "tmd_stage2", "occupancy_tmd", "occupancy_discriminator"])
def test_removed_method_tags_are_rejected(tmp_path: Path, removed: str) -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "configs/methods/dmd2_flow.yaml").read_text()
    path = tmp_path / f"{removed}.yaml"
    path.write_text(text.replace("method: dmd2_flow", f"method: {removed}", 1))
    with pytest.raises(ConfigError, match="unknown production method"):
        load_config(path)


def test_deprecated_dmd_fields_are_rejected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    original = (root / "configs/methods/dmd2_flow.yaml").read_text()
    path = tmp_path / "legacy.yaml"
    path.write_text(original.replace("  gan_weight: 0.003", "  gan_weight: 0.003\n  data_weight: 1.0"))
    with pytest.raises(ConfigError, match="unknown or deprecated"):
        load_config(path)


def test_cli_exposes_only_retained_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["train", "dmd2-flow"]).train_method == "dmd2-flow"
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "tmd-stage1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["evaluate", "compare"])


def test_sampled_evaluation_cli_accepts_checkpoint_scope() -> None:
    args = build_parser().parse_args([
        "evaluate", "libero", "--checkpoint", "step.pt", "--checkpoint-sha256", "auto",
        "--device", "cuda:2", "--suite", "libero_spatial", "--task-ids", "0", "5",
        "--reset-seeds", "0", "--max-episode-steps", "20",
    ])
    assert args.suite == ["libero_spatial"] and args.task_ids == [0, 5]


def test_inference_delta_loads_only_declared_parameters() -> None:
    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(2, 1)
            self.frozen = nn.Linear(2, 2)

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = Policy()
            self.trainable_parameter_names = ("model.weight", "model.bias")

    student = Student()
    frozen = student.policy.frozen.weight.detach().clone()
    delta = {
        "student.policy.model.weight": torch.full_like(student.policy.model.weight, 3.0),
        "student.policy.model.bias": torch.full_like(student.policy.model.bias, -2.0),
    }
    _load_student_checkpoint_state(student, delta, checkpoint_format="tmdpolicy.inference/v1", trainable_parameter_names=list(delta))
    assert torch.equal(student.policy.model.weight, delta["student.policy.model.weight"])
    assert torch.equal(student.policy.frozen.weight, frozen)


def test_dmd2_inference_state_contains_only_student_delta() -> None:
    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy = nn.Module()
            self.policy.model = nn.Linear(2, 1)
            self.trainable_parameter_names = ("model.weight", "model.bias")

    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    program.student = Student()
    assert set(program.inference_state_dict()) == {
        "student.policy.model.weight", "student.policy.model.bias"
    }


def test_existing_dmd2_final_checkpoint_metadata_is_compatible() -> None:
    root = Path(__file__).resolve().parents[1]
    inference = root / "artifacts/training/dmd2_final/inference_checkpoints/step-00001000.pt"
    full = root / "artifacts/training/dmd2_final/checkpoints/step-00001000.pt"
    if not inference.exists() or not full.exists():
        pytest.skip("local dmd2_final artifacts are not present")
    inference_payload = torch.load(inference, map_location="cpu", weights_only=False, mmap=True)
    assert inference_payload["format"] == "tmdpolicy.inference/v1"
    assert inference_payload["config"]["method"] == "dmd2_flow"
    assert set(inference_payload["program"]) == set(inference_payload["trainable_parameter_names"])
    full_payload = torch.load(full, map_location="cpu", weights_only=False, mmap=True)
    assert full_payload["format"] == "tmdpolicy.training/v1"
    assert {key.split(".", 1)[0] for key in full_payload["program"]} == {
        "student", "fake_score", "discriminator", "bridge"
    }
    assert set(full_payload["optimizers"]) == {"guidance", "generator"}


def test_pi05_evaluation_does_not_construct_smolvla(monkeypatch) -> None:
    import tmd_policy.evaluation.policy as module

    class Teacher:
        device = torch.device("cpu")
        model_id, model_revision, processor_revision = "pi05", "a" * 40, "b" * 40
        policy = SimpleNamespace(config=SimpleNamespace(num_inference_steps=10))
        preprocess_observation = lambda self, batch: batch
        encode_condition = lambda self, batch: SimpleNamespace(batch_size=1)
        sample = lambda self, condition, noise, num_steps, step_callback=None: noise
        postprocess_action = lambda self, value: value[..., :7]

    monkeypatch.setattr(module, "build_teacher", lambda config, device: Teacher())
    monkeypatch.setattr(module, "build_student", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    policy, identity = module.load_inference_policy({"policy": {"method": "pi05", "device": "cpu", "num_steps": 10}})
    assert policy.plan({"observation.state": torch.zeros(1, 8)}, noise_seed=7).shape == (1, 50, 7)
    assert identity["num_steps"] == 10


def test_smolvla_official_mode_uses_checkpoint_ten_steps(monkeypatch) -> None:
    import tmd_policy.evaluation.policy as module

    projection = nn.Identity()

    class Official:
        config = SimpleNamespace(num_steps=10)
        def predict_action_chunk(self, batch, *, noise):
            for _ in range(10): projection(noise)
            return noise
        def reset(self): pass

    class Student:
        device = torch.device("cpu")
        model_id, model_revision, processor_revision = "smol", "c" * 40, "d" * 40
        policy, flow = Official(), SimpleNamespace(action_out_proj=projection)
        preprocess_observation = lambda self, batch: batch
        postprocessor = lambda self, value: value[..., :7]

    monkeypatch.setattr(module, "build_student", lambda config, device: Student())
    policy, identity = module.load_inference_policy({"policy": {"method": "smolvla", "device": "cpu", "sampler_mode": "official"}})
    completed = []
    plan = policy.plan({"observation.state": torch.zeros(1, 8)}, noise_seed=13, step_callback=lambda: completed.append(1))
    assert plan.shape == (1, 50, 7) and len(completed) == 10
    assert identity["num_steps"] == 10
