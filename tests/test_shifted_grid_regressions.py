from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent
from tmd_policy.methods.dmd2_flow.program import DMD2FlowProgram
from tmd_policy.methods.flow_objectives import shifted_time_grid
from tmd_policy.methods.tmd.meanflow import meanflow_loss, sample_meanflow_batch
from tmd_policy.methods.tmd.program import sample_stage1_generator
from tmd_policy.training.engine import validate_optimizer_parameter_ownership


def test_tmd_outer_times_and_predecessors_use_shifted_student_grids() -> None:
    reference = torch.zeros(4096, 1, 1)
    samples = sample_meanflow_batch(
        reference,
        flow_matching_fraction=0.0,
        student_time_shift_gamma=10.0,
        meanflow_time_shift_gamma=3.0,
        discrete_outer_steps=4,
        discrete_inner_steps=7,
        generator=torch.Generator().manual_seed(9),
    )
    outer_grid = shifted_time_grid(4, 10.0, device=reference.device)
    inner_grid = shifted_time_grid(7, 10.0, device=reference.device)
    assert torch.all(torch.isin(samples.outer_time, outer_grid[1:]))
    expected_indices = torch.searchsorted(
        inner_grid, samples.inner_time.contiguous(), right=True
    ) - 1
    assert torch.equal(samples.target_time, inner_grid[expected_indices])
    assert torch.all(samples.target_time <= samples.inner_time)


class _RecordingStudent:
    def __init__(self) -> None:
        self.times: list[torch.Tensor] = []

    def velocity_with_features(self, condition, value, time):
        self.times.append(time.detach().clone())
        return torch.zeros_like(value), torch.zeros(*value.shape[:-1], 3)


class _RecordingHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.intervals: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, value, s, r, context):
        self.intervals.append((s.detach().clone(), r.detach().clone()))
        return torch.zeros_like(value)


def test_tmd_training_sampler_and_both_inference_loops_share_shifted_grid() -> None:
    student = _RecordingStudent()
    head = _RecordingHead()
    noise = torch.zeros(2, 5, 4)
    gamma = 10.0
    completed_work = []
    sample_stage1_generator(
        student,
        head,
        SimpleNamespace(batch_size=2),
        noise,
        outer_steps=3,
        inner_steps=2,
        student_time_shift_gamma=gamma,
        inner_noises=torch.zeros(3, 2, 5, 4),
        step_callback=lambda: completed_work.append(1),
    )
    assert len(completed_work) == 3 * (2 + 1)
    outer = shifted_time_grid(3, gamma, device=noise.device, descending=True)
    inner = shifted_time_grid(2, gamma, device=noise.device, descending=True)
    assert torch.allclose(torch.stack([value[0] for value in student.times]), outer[:-1])
    expected_intervals = list(zip(inner[:-1], inner[1:], strict=True)) * 3
    for (actual_s, actual_r), (expected_s, expected_r) in zip(
        head.intervals, expected_intervals, strict=True
    ):
        assert torch.allclose(actual_s, expected_s.expand(2))
        assert torch.allclose(actual_r, expected_r.expand(2))


class _ZeroHead(nn.Module):
    def forward(self, value, s, r, context):
        return torch.zeros_like(value)


def test_adaptive_meanflow_normalization_uses_valid_squared_norm_and_count() -> None:
    target = torch.zeros(2, 50, 7)
    source = torch.ones_like(target)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[1, 40:] = False
    result = meanflow_loss(
        _ZeroHead(),
        target_transition=target,
        inner_source=source,
        inner_time=torch.tensor([0.25, 0.75]),
        target_time=torch.tensor([0.0, 0.5]),
        context=torch.zeros(2, 50, 1),
        valid_coordinates=valid,
        normalization_constant_scale=1.0,
    )
    expected_count = torch.tensor([350.0, 280.0])
    assert torch.equal(result["valid_coordinate_count"], expected_count)
    assert torch.equal(result["normalization_constant"], expected_count)
    expected = result["squared_error_sum"] / (
        result["squared_error_sum"] + expected_count
    )
    assert torch.allclose(result["per_sample_loss"], expected)


def test_optimizer_parameter_ownership_rejects_cross_optimizer_overlap() -> None:
    shared = nn.Parameter(torch.tensor(1.0))
    first = torch.optim.AdamW([shared], lr=1.0e-3)
    second = torch.optim.AdamW([shared], lr=1.0e-3)
    with pytest.raises(RuntimeError, match="optimizer parameter overlap"):
        validate_optimizer_parameter_ownership({"first": first, "second": second})


def test_guidance_backward_updates_fake_and_classifier_but_not_student_or_teacher() -> None:
    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    program.feature_source = "fake_score_features"
    program.dmd_config = {"guidance_classifier_weight": 2.0}
    program.fake_score = nn.Linear(1, 1, bias=False)
    program.discriminator = nn.Linear(1, 1, bias=False)
    program.student = nn.Linear(1, 1, bias=False)
    program.teacher = nn.Linear(1, 1, bias=False)
    calls: list[tuple[dict, torch.Tensor, object]] = []
    conditions: list[dict] = []
    expert_batch = {"source": "expert"}
    replay_batch = {"source": "replay", "_student_replay_batch": True}

    def sample(self, batch, *, requires_grad):
        assert not requires_grad
        marker = 1.0 if batch is replay_batch else 2.0
        return torch.full((1, 2, 1), marker)

    def teacher_condition(self, batch):
        conditions.append(batch)
        return batch

    def fake(self, batch, generated=None, *, teacher_condition=None):
        assert batch is replay_batch and teacher_condition is replay_batch
        calls.append((batch, generated, teacher_condition))
        return self.fake_score.weight.square().sum(), {"denoising": 1.0}

    def classifier(self, batch, generated=None, *, teacher_condition=None):
        assert batch is expert_batch and teacher_condition is expert_batch
        calls.append((batch, generated, teacher_condition))
        return self.discriminator.weight.square().sum(), {"classification": 1.0}

    program._sample_student = MethodType(sample, program)
    program._teacher_condition = MethodType(teacher_condition, program)
    program._fake_loss = MethodType(fake, program)
    program._discriminator_loss = MethodType(classifier, program)
    loss, metrics = program._guidance_loss(
        {
            "_dmd2_guidance_expert_batch": expert_batch,
            "_dmd2_guidance_fake_score_batch": replay_batch,
        }
    )
    loss.backward()
    assert len(calls) == 2
    assert calls[0][0] is replay_batch and calls[1][0] is expert_batch
    assert not calls[0][1].requires_grad and not calls[1][1].requires_grad
    assert calls[0][1].unique().item() == 1.0
    assert calls[1][1].unique().item() == 2.0
    assert program.fake_score.weight.grad is not None
    assert program.discriminator.weight.grad is not None
    assert program.student.weight.grad is None
    assert program.teacher.weight.grad is None
    assert conditions == [replay_batch, expert_batch]
    assert metrics["fake_score_replay_state_fraction"] == 1.0
    assert metrics["gan_expert_state_fraction"] == 1.0


def test_dmd2_phase_batch_routes_fake_score_to_replay_and_gan_to_expert() -> None:
    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    expert_batch = {"observation.state": torch.zeros(2, 8)}
    replay_batch = {
        "observation.state": torch.ones(2, 8),
        "_student_replay_batch": True,
    }

    class Replay:
        def sample_like(self, batch):
            assert batch is expert_batch
            return replay_batch

    program._student_replay = Replay()
    guidance = program.prepare_phase_batch(expert_batch, "guidance")
    assert guidance["_dmd2_guidance_expert_batch"] is expert_batch
    assert guidance["_dmd2_guidance_fake_score_batch"] is replay_batch
    assert program.prepare_phase_batch(expert_batch, "fake") is replay_batch
    assert program.prepare_phase_batch(expert_batch, "generator") is replay_batch
    assert program.prepare_phase_batch(expert_batch, "discriminator") is expert_batch


def test_dmd2_runtime_gradient_contract_keeps_teacher_and_backbone_frozen() -> None:
    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    program.teacher = SimpleNamespace(policy=nn.Linear(2, 2))
    program.teacher.policy.requires_grad_(False)
    program.fake_score = nn.Linear(2, 2)
    program.discriminator = nn.Linear(2, 1)
    program.student = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    program.student[0].requires_grad_(False)

    program.fake_score.weight.grad = torch.ones_like(program.fake_score.weight)
    program.discriminator.weight.grad = torch.ones_like(program.discriminator.weight)
    program.validate_phase_gradients("guidance")
    assert all(parameter.grad is None for parameter in program.teacher.policy.parameters())
    assert all(parameter.grad is None for parameter in program.student[0].parameters())

    program.student[1].weight.grad = torch.ones_like(program.student[1].weight)
    program.validate_phase_gradients("generator")
    program.teacher.policy.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="teacher must be completely frozen"):
        program.validate_phase_gradients("generator")


class _DMDStudent(nn.Module):
    chunk_size = 2
    internal_action_dim = 1

    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))
        self.policy = SimpleNamespace(
            prepare_action=lambda batch: (_ for _ in ()).throw(
                AssertionError("backward simulation must not read expert actions")
            )
        )
        self.times: list[torch.Tensor] = []

    def preprocess_observation(self, batch):
        return batch

    def encode_condition(self, batch):
        return SimpleNamespace(batch_size=batch["observation.state"].shape[0])

    def velocity(self, condition, value, time):
        self.times.append(time.detach().clone())
        return torch.full_like(value, 2.0) + self.anchor * 0.0

    clean_prediction = staticmethod(LeRobotSmolVLAStudent.clean_prediction)


def test_dmd2_backward_simulation_and_inference_share_denoise_renoise_grid(monkeypatch) -> None:
    student = _DMDStudent()
    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    program.student = student
    program.dmd_config = {"discrete_outer_steps": 4, "student_time_shift_gamma": 5.0}
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, *, device=None, dtype=None, generator=None: torch.ones(
            shape, device=device, dtype=dtype
        ),
    )
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, device=None, generator=None: torch.full(
            size, 2, device=device, dtype=torch.long
        ),
    )
    clean = torch.full((1, 2, 1), 3.0)
    predicted = program._sample_student(
        {"observation.state": torch.zeros(1, 8)}, requires_grad=True
    )
    grid = shifted_time_grid(4, 5.0, device=clean.device, descending=True)
    value = torch.ones_like(clean)
    for index in range(2):
        clean_prediction = value - grid[index] * 2.0
        value = (1.0 - grid[index + 1]) * clean_prediction + grid[index + 1]
    assert torch.allclose(predicted, value - grid[2] * 2.0)
    assert torch.allclose(torch.stack([value[0] for value in student.times]), grid[:3])
    assert program._last_backward_metrics["backward/selected_step_index"] == 2.0

    student.times.clear()
    LeRobotSmolVLAStudent.sample_denoise_renoise(
        student,
        SimpleNamespace(batch_size=1),
        torch.ones_like(clean),
        4,
        student_time_shift_gamma=5.0,
        renoise_noises=torch.ones(3, *clean.shape),
    )
    assert torch.allclose(torch.stack([value[0] for value in student.times]), grid[:-1])
