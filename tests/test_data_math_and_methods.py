from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tmd_policy.backends.lerobot.smolvla_student import LeRobotSmolVLAStudent
from tmd_policy.data.libero import assert_episode_disjoint, stratified_episode_split
from tmd_policy.methods.dmd2_flow.program import DMD2FlowProgram


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


class _ToyFlow(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.state_proj = nn.Linear(1, 2)
        self.action_in_proj = nn.Linear(2, 2)
        self.action_out_proj = nn.Linear(2, 2)
        self.action_time_mlp_in = nn.Linear(2, 2)
        self.action_time_mlp_out = nn.Linear(2, 2)
        self.vlm_with_expert = nn.Module()
        self.vlm_with_expert.lm_expert = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        self.vlm_with_expert.vlm = nn.Linear(2, 2)


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


def test_smolvla_action_expert_mode_unfreezes_expert_and_heads_only() -> None:
    student = _toy_student()
    head_names = student.configure_trainable("head_only")
    assert head_names == (
        "model.action_in_proj.weight",
        "model.action_in_proj.bias",
        "model.action_out_proj.weight",
        "model.action_out_proj.bias",
        "model.action_time_mlp_in.weight",
        "model.action_time_mlp_in.bias",
        "model.action_time_mlp_out.weight",
        "model.action_time_mlp_out.bias",
    )
    names = student.configure_trainable("action_expert")
    assert any("vlm_with_expert.lm_expert" in name for name in names)
    assert set(head_names) < set(names)
    assert not any("vlm_with_expert.vlm" in name for name in names)
    assert not student.policy.model.state_proj.weight.requires_grad
    expert_parameters = sum(
        parameter.numel()
        for name, parameter in student.policy.named_parameters()
        if name in names
    )
    head_parameters = sum(
        parameter.numel()
        for name, parameter in student.policy.named_parameters()
        if name in head_names
    )
    assert expert_parameters > head_parameters
    with pytest.raises(ValueError, match="action_expert.*head_only"):
        student.configure_trainable("full")  # type: ignore[arg-type]


def test_dmd2_ttur_phase_schedule_uses_five_guidance_updates() -> None:
    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    program.dmd_config = {"fake_updates_per_generator": 5}
    assert program.phase_schedule() == ("guidance",) * 5 + ("generator",)


def test_dmd2_phase_schedule_rejects_invalid_ratio() -> None:
    program = DMD2FlowProgram.__new__(DMD2FlowProgram)
    nn.Module.__init__(program)
    program.dmd_config = {"fake_updates_per_generator": 0}
    with pytest.raises(ValueError, match="positive"):
        program.phase_schedule()
