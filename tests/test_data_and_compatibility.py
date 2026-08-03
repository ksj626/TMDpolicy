
import numpy as np
import torch
from torch import nn

from tmd_policy.compatibility.actions import PolicyActionBridge, StateCompatibilityAdapter
from tmd_policy.data.expert import episode_split, episode_split_three_way
from tmd_policy.data.schemas import TeacherQuery
from tmd_policy.teacher.cache import TeacherQueryCache
from tmd_policy.teacher.query import FrozenTeacherQuerier


class Preprocessor:
    def __call__(self, batch):
        result = dict(batch)
        result["action"] = (batch["action"] - 0.2) / 0.5
        return result


class Postprocessor:
    def __call__(self, action):
        return action * 0.5 + 0.2


def test_action_bridge_roundtrip_uses_policy_specific_statistics():
    bridge = PolicyActionBridge(Preprocessor(), Postprocessor())
    actions = torch.rand(2, 50, 7) * 1.5 - 0.75
    assert bridge.round_trip_error(actions) < 1e-6


def test_canonical_state_adapter_is_explicit():
    state = torch.arange(8).float()
    adapter = StateCompatibilityAdapter()
    assert adapter.for_student(state).tolist() == list(range(8))
    assert adapter.for_teacher(state).tolist() == list(range(8))


def test_episode_split_never_leaks_an_episode():
    mapping = {index: index // 5 for index in range(20)}
    train, held_out = episode_split(mapping, 0.2, 1)
    assert set(train).isdisjoint(held_out)
    assert set(train) | set(held_out) == set(mapping)


def test_three_way_episode_split_is_task_stratified_and_disjoint():
    mapping = {index: index // 6 for index in range(24)}
    train, validation, test = episode_split_three_way(mapping, 0.2, 0.2, seed=3)
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    assert set(train) | set(validation) | set(test) == set(mapping)
    for task in set(mapping.values()):
        task_episodes = {episode for episode, value in mapping.items() if value == task}
        assert task_episodes & set(train)
        assert task_episodes & set(validation)
        assert task_episodes & set(test)


def _query(sample_id="q", seed=1, sample_index=0):
    return TeacherQuery(
        sample_id=sample_id,
        observation_id="obs",
        task_index=0,
        instruction="task",
        teacher_checkpoint="pi05",
        teacher_revision="revision",
        processor_revision="processor-revision",
        sampling_seed=seed,
        inference_steps=10,
        sample_index=sample_index,
        action_chunk=np.zeros((50, 7)),
        action_valid=np.ones(50, dtype=bool),
    )


def test_store_preserves_teacher_provenance_and_multiple_modes(tmp_path):
    cache = TeacherQueryCache(tmp_path)
    cache.put(_query("q1", seed=1, sample_index=0))
    cache.put(_query("q2", seed=2, sample_index=1))
    assert len(cache.store) == 2
    record, arrays = cache.get(
        "obs", "pi05", "revision", "processor-revision", 2, 10, 1
    )
    assert record["sampling_seed"] == 2
    assert arrays["action_chunk"].shape == (50, 7)


class FakeTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0
        self.config = type("Config", (), {"num_inference_steps": 99})()

    def predict_action_chunk(self, batch):
        self.calls += 1
        return torch.full((1, 50, 7), 1.5)


def test_frozen_teacher_query_is_clipped_and_cached(tmp_path):
    policy = FakeTeacher()
    cache = TeacherQueryCache(tmp_path)
    querier = FrozenTeacherQuerier(
        policy,
        lambda value: value,
        lambda value: value,
        cache,
        checkpoint="teacher",
        revision="rev",
        processor_revision="processor-rev",
    )
    kwargs = {
        "observation_id": "obs",
        "task_index": 0,
        "instruction": "task",
        "sampling_seed": 3,
    }
    first = querier.query({}, **kwargs)
    second = querier.query({}, **kwargs)
    assert policy.calls == 1
    assert policy.config.num_inference_steps == 10
    assert np.array_equal(first, second)
    assert first.max() == 1.0
