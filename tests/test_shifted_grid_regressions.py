from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent
from tmd_policy.methods.denoise_renoise import denoise_renoise_prefix, denoise_renoise_sample
from tmd_policy.methods.dmd2_flow.program import DMD2FlowProgram
from tmd_policy.methods.flow_objectives import shifted_time_grid
from tmd_policy.training.engine import validate_optimizer_parameter_ownership


def test_optimizer_parameter_ownership_rejects_overlap() -> None:
    shared = nn.Parameter(torch.tensor(1.0))
    with pytest.raises(RuntimeError, match="optimizer parameter overlap"):
        validate_optimizer_parameter_ownership(
            {
                "guidance": torch.optim.AdamW([shared], lr=1.0e-3),
                "generator": torch.optim.AdamW([shared], lr=1.0e-3),
            }
        )


def test_guidance_updates_fake_and_classifier_not_student_or_teacher() -> None:
    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    program.feature_source = "fake_score_features"
    program.dmd_config = {"guidance_classifier_weight": 2.0}
    program.fake_score = nn.Linear(1, 1, bias=False)
    program.discriminator = nn.Linear(1, 1, bias=False)
    program.student = nn.Linear(1, 1, bias=False)
    program.teacher = nn.Linear(1, 1, bias=False)
    expert = {"source": "expert"}
    replay = {"source": "replay", "_student_replay_batch": True}

    def sample(self, batch, *, requires_grad):
        assert not requires_grad
        return torch.full((1, 2, 1), 1.0 if batch is replay else 2.0)

    program._sample_student = MethodType(sample, program)
    program._teacher_condition = MethodType(lambda self, batch: batch, program)
    program._fake_loss = MethodType(
        lambda self, batch, generated=None, teacher_condition=None: (
            self.fake_score.weight.square().sum(),
            {},
        ),
        program,
    )
    program._discriminator_loss = MethodType(
        lambda self, batch, generated=None, teacher_condition=None: (
            self.discriminator.weight.square().sum(),
            {},
        ),
        program,
    )
    loss, metrics = program._guidance_loss(
        {
            "_dmd2_guidance_expert_batch": expert,
            "_dmd2_guidance_fake_score_batch": replay,
        }
    )
    loss.backward()
    assert program.fake_score.weight.grad is not None
    assert program.discriminator.weight.grad is not None
    assert program.student.weight.grad is None
    assert program.teacher.weight.grad is None
    assert metrics["fake_score_replay_state_fraction"] == 1.0
    assert metrics["gan_expert_state_fraction"] == 1.0


def test_phase_batch_routes_score_to_replay_and_gan_to_expert() -> None:
    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    expert = {"observation.state": torch.zeros(2, 8)}
    replay = {"observation.state": torch.ones(2, 8), "_student_replay_batch": True}

    class Replay:
        def sample_like(self, batch):
            assert batch is expert
            return replay

    program._student_replay = Replay()
    guidance = program.prepare_phase_batch(expert, "guidance")
    assert guidance["_dmd2_guidance_expert_batch"] is expert
    assert guidance["_dmd2_guidance_fake_score_batch"] is replay
    assert program.prepare_phase_batch(expert, "generator") is replay


def test_gradient_contract_keeps_teacher_and_backbone_frozen() -> None:
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


def test_shared_denoise_renoise_prefix_matches_full_sampler() -> None:
    grid = shifted_time_grid(4, 5.0, device=torch.device("cpu"), descending=True)
    noise = torch.ones(1, 2, 1)
    fresh = torch.stack((torch.full_like(noise, 2), torch.full_like(noise, 3), torch.full_like(noise, 4)))
    velocity = lambda value, time: torch.full_like(value, 0.5)
    prefix = denoise_renoise_prefix(
        velocity,
        noise,
        grid,
        2,
        renoise_noises=fresh[:2],
    )
    expected = noise
    for index in range(2):
        clean = expected - grid[index] * 0.5
        expected = (1 - grid[index + 1]) * clean + grid[index + 1] * fresh[index]
    assert torch.allclose(prefix, expected)
    result = denoise_renoise_sample(velocity, noise, grid, renoise_noises=fresh)
    value = noise
    for index in range(4):
        clean = value - grid[index] * 0.5
        if index < 3:
            value = (1 - grid[index + 1]) * clean + grid[index + 1] * fresh[index]
    assert torch.allclose(result, clean)


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


def test_training_backward_simulation_and_inference_share_grid(monkeypatch) -> None:
    student = _DMDStudent()
    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    program.student = student
    program.dmd_config = {"discrete_outer_steps": 4, "student_time_shift_gamma": 5.0}
    monkeypatch.setattr(
        torch,
        "randn",
        lambda shape, *, device=None, dtype=None, generator=None: torch.ones(shape, device=device, dtype=dtype),
    )
    monkeypatch.setattr(
        torch,
        "randint",
        lambda low, high, size, device=None, generator=None: torch.full(size, 2, device=device, dtype=torch.long),
    )
    predicted = program._sample_student({"observation.state": torch.zeros(1, 8)}, requires_grad=True)
    grid = shifted_time_grid(4, 5.0, device=torch.device("cpu"), descending=True)
    value = torch.ones(1, 2, 1)
    for index in range(2):
        clean = value - grid[index] * 2.0
        value = (1 - grid[index + 1]) * clean + grid[index + 1]
    assert torch.allclose(predicted, value - grid[2] * 2.0)
    assert torch.allclose(torch.stack([value[0] for value in student.times]), grid[:3])
    assert program._last_backward_metrics["backward/selected_step_index"] == 2.0

    student.times.clear()
    LeRobotSmolVLAStudent.sample_denoise_renoise(
        student,
        SimpleNamespace(batch_size=1),
        torch.ones(1, 2, 1),
        4,
        student_time_shift_gamma=5.0,
        renoise_noises=torch.ones(3, 1, 2, 1),
    )
    assert torch.allclose(torch.stack([value[0] for value in student.times]), grid[:-1])
