from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch import nn

from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent
from tmd_policy.data.libero import assert_episode_disjoint, stratified_episode_split
from tmd_policy.evaluation.compare import compare
from tmd_policy.methods.dmd2_flow.program import DMD2FlowProgram
from tmd_policy.methods.occupancy_tmd.program import weighted_generator_loss
from tmd_policy.methods.occupancy_tmd.networks import WindowNormalizer
from tmd_policy.methods.occupancy_tmd.program import OccupancyDiscriminatorProgram
from tmd_policy.methods.tmd.heads import GRUMeanFlowHead, SplitTransformerMeanFlowHead
from tmd_policy.methods.tmd.meanflow import meanflow_loss, sample_meanflow_batch


def test_episode_split_is_task_stratified_and_disjoint() -> None:
    episode_to_task = {episode: episode // 10 for episode in range(40)}
    splits = stratified_episode_split(
        episode_to_task, validation_fraction=0.2, test_fraction=0.2, seed=17
    )
    assert_episode_disjoint(splits)
    assert set().union(*map(set, splits.values())) == set(episode_to_task)
    for task in range(4):
        for split in splits.values():
            assert any(episode_to_task[episode] == task for episode in split)


def test_lerobot_terminal_query_mask_marks_repeated_boundary() -> None:
    from lerobot.datasets.dataset_reader import DatasetReader

    reader = DatasetReader.__new__(DatasetReader)
    reader._meta = SimpleNamespace(episodes={0: {"dataset_from_index": 0, "dataset_to_index": 3}})
    reader.delta_indices = {"action": [0, 1, 2, 3]}
    indices, padding = reader._get_query_indices(1, 0)
    assert indices["action"] == [1, 2, 2, 2]
    assert padding["action_is_pad"].tolist() == [False, False, True, True]


def test_tmd_gaussian_sources_are_independent_and_rs_mixture_is_real() -> None:
    reference = torch.zeros(20_000, 2, 2)
    samples = sample_meanflow_batch(reference, flow_matching_fraction=0.73)
    correlation = torch.corrcoef(
        torch.stack((samples.outer_noise.flatten(), samples.inner_source.flatten()))
    )[0, 1]
    assert abs(float(correlation)) < 0.02
    assert abs(float(samples.flow_matching_rows.float().mean()) - 0.73) < 0.015
    assert torch.all(samples.target_time <= samples.inner_time)
    assert torch.equal(samples.target_time[samples.flow_matching_rows], samples.inner_time[samples.flow_matching_rows])


def test_meanflow_target_is_stopped_and_valid_mask_applies() -> None:
    head = GRUMeanFlowHead(action_dim=2, context_dim=3, hidden_dim=8, layers=1)
    transition = torch.randn(3, 4, 2)
    source = torch.randn_like(transition)
    valid = torch.ones_like(transition, dtype=torch.bool)
    valid[:, -1] = False
    values = meanflow_loss(
        head,
        target_transition=transition,
        inner_source=source,
        inner_time=torch.tensor([0.2, 0.5, 0.9]),
        target_time=torch.tensor([0.1, 0.5, 0.2]),
        context=torch.randn(3, 4, 3),
        valid_coordinates=valid,
        normalization_constant=1.0,
    )
    assert not values["target"].requires_grad
    values["loss"].backward()
    assert any(parameter.grad is not None for parameter in head.parameters())


def test_primary_split_transformer_supports_meanflow_jvp_and_backward() -> None:
    head = SplitTransformerMeanFlowHead(
        action_dim=4,
        context_dim=6,
        model_dim=16,
        layers=2,
        heads=4,
        feedforward_dim=32,
        horizon=5,
    )
    target = torch.randn(2, 5, 4)
    values = meanflow_loss(
        head,
        target_transition=target,
        inner_source=torch.randn_like(target),
        inner_time=torch.tensor([0.4, 0.8]),
        target_time=torch.tensor([0.1, 0.8]),
        context=torch.randn(2, 5, 6),
        valid_coordinates=torch.ones_like(target, dtype=torch.bool),
        normalization_constant=1.0,
    )
    values["loss"].backward()
    assert torch.isfinite(values["total_derivative"]).all()
    assert any(parameter.grad is not None for parameter in head.parameters())


class _ToyFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_proj = nn.Linear(1, 2)
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.action_time_mlp_in = nn.Linear(2, 2)
        self.action_time_mlp_out = nn.Linear(2, 2)
        self.vlm_with_expert = _ToyExpertContainer()


class _ToyExpertContainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lm_expert = nn.Sequential(nn.Linear(2, 2))


class _ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ToyFlow()


def _toy_student() -> LeRobotSmolVLAStudent:
    student = LeRobotSmolVLAStudent.__new__(LeRobotSmolVLAStudent)
    nn.Module.__init__(student)
    student.policy = _ToyPolicy()
    student.trainable_parameter_names = ()
    return student


def test_flow_sft_explicit_module_selection() -> None:
    student = _toy_student()
    head_names = student.configure_trainable("head_only")
    assert head_names
    assert all("lm_expert" not in name for name in head_names)
    expert_names = student.configure_trainable("expert_only")
    assert any("lm_expert" in name for name in expert_names)


def test_dmd2_ttur_and_shared_sampler_call() -> None:
    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    program.dmd_config = {"fake_updates_per_generator": 4, "generation_steps": 3}

    class SpyStudent(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.tensor(0.0))
            self.calls = 0
            self.device = torch.device("cpu")

        def preprocess_observation(self, batch):
            return batch

        def encode_condition(self, batch):
            return SimpleNamespace(batch_size=2)

        def sample(self, condition, noise, steps):
            self.calls += 1
            assert steps == 3
            return noise + self.anchor

    program.student = SpyStudent()
    value = program._sample_student({}, requires_grad=True)
    assert value.shape == (2, 50, 32)
    assert program.student.calls == 1
    assert program.phase_schedule() == ("fake", "fake", "fake", "fake", "discriminator", "generator")


def test_occupancy_weights_change_generator_gradient() -> None:
    parameter = torch.tensor([1.0, -1.0], requires_grad=True)
    per_sample = parameter.square()
    weighted_generator_loss(per_sample, torch.tensor([1.0, 1.0])).backward()
    equal_gradient = parameter.grad.detach().clone()
    parameter.grad = None
    weighted_generator_loss(parameter.square(), torch.tensor([0.5, 2.0])).backward()
    assert not torch.allclose(equal_gradient, parameter.grad)


def test_occupancy_discriminator_orientation_is_expert_one_student_zero() -> None:
    class OrientedDiscriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, state, action, visual, task_index, position, valid):
            return state[:, 0, 0] * self.scale

    normalizer = WindowNormalizer(
        torch.zeros(8),
        torch.ones(8),
        torch.zeros(7),
        torch.ones(7),
        torch.zeros(6),
        torch.ones(6),
        fitted_samples=2,
    )
    program = OccupancyDiscriminatorProgram(OrientedDiscriminator(), normalizer)
    state = torch.zeros(2, 3, 8)
    state[0, :, 0] = 4.0
    state[1, :, 0] = -4.0
    loss, metrics = program.loss(
        {
            "state": state,
            "action": torch.zeros(2, 3, 7),
            "visual": torch.zeros(2, 3, 6),
            "task_index": torch.zeros(2, dtype=torch.long),
            "position": torch.arange(3).expand(2, -1),
            "valid": torch.ones(2, 3, dtype=torch.bool),
            "source_label": torch.tensor([1.0, 0.0]),
            "balance_weight": torch.ones(2),
        },
        "discriminator",
    )
    assert float(loss.detach()) < 0.05
    assert metrics["expert_probability"] > metrics["student_probability"]


def test_paired_comparison_reports_overall_suite_and_task_results(tmp_path: Path) -> None:
    episodes = [
        {"suite": suite, "task_id": task, "reset_seed": seed, "success": (task + seed) % 2 == 0}
        for suite in ("libero_goal", "libero_object")
        for task in (0, 1)
        for seed in (0, 1)
    ]
    baseline = tmp_path / "baseline.json"
    method = tmp_path / "method.json"
    baseline.write_text(json.dumps({"episodes": episodes}), encoding="utf-8")
    improved = [dict(row, success=True) for row in episodes]
    method.write_text(json.dumps({"episodes": improved}), encoding="utf-8")
    config = tmp_path / "comparison.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "inputs": {"baseline": str(baseline), "method": str(method)},
                "statistics": {
                    "pairing_keys": ["suite", "task_id", "reset_seed"],
                    "bootstrap_resamples": 100,
                    "confidence": 0.95,
                    "seed": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    report = compare(config, tmp_path / "output")
    assert report["pair_count"] == 8
    assert set(report["per_suite"]) == {"libero_goal", "libero_object"}
    assert set(report["per_task"]) == {
        "libero_goal:0",
        "libero_goal:1",
        "libero_object:0",
        "libero_object:1",
    }
    assert report["comparisons"]["method"]["difference_from_baseline"] == 0.5
