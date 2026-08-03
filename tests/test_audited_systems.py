from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from tmd_policy.config import config_runtime_report, load_config, render_config_reference
from tmd_policy.data.schemas import ExpertChunk, TeacherQuery
from tmd_policy.data.storage import ChunkStore, StorageIntegrityError
from tmd_policy.evaluation.metrics import bootstrap_episode_statistic
from tmd_policy.models.discriminator import PathNormalizer
from tmd_policy.rollout.collector import (
    CanonicalChunkRunner,
    PlanResult,
    collect_rollout_episode,
)
from tmd_policy.training.checkpoint import load_training_checkpoint, save_training_checkpoint
from tmd_policy.training.distillation import combined_distillation_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _expert(sample_id: str = "expert-1") -> ExpertChunk:
    return ExpertChunk(
        sample_id=sample_id,
        observation_id="observation-1",
        dataset_id="dataset",
        dataset_revision="revision",
        episode_index=0,
        task_index=0,
        instruction="pick up the object",
        start_frame=0,
        plan_actions=np.zeros((50, 7), dtype=np.float32),
        plan_valid=np.ones(50, dtype=bool),
        path_states=np.zeros((11, 8), dtype=np.float32),
        path_actions=np.zeros((10, 7), dtype=np.float32),
        path_valid=np.ones(10, dtype=bool),
    )


def test_unknown_nested_config_key_is_rejected(tmp_path):
    raw = yaml.safe_load((PROJECT_ROOT / "configs" / "tiny.yaml").read_text())
    raw["tmd"]["inner_soruce_mode"] = "gaussian_tm"
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(KeyError, match="inner_soruce_mode"):
        load_config(path)


def test_every_config_leaf_has_a_consumer_and_generated_docs_match():
    report = config_runtime_report()
    assert report and all(report.values())
    reference = PROJECT_ROOT / "docs" / "config_reference.md"
    assert reference.read_text(encoding="utf-8") == render_config_reference()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("plan_actions", np.zeros((49, 7)), "expected shape"),
        ("path_states", np.zeros((10, 8)), "expected shape"),
        ("plan_valid", np.array([True, False] + [True] * 48), "prefix-contiguous"),
        ("path_actions", np.full((10, 7), 1.1), "bounds"),
    ],
)
def test_expert_schema_rejects_contract_violations(field, value, message):
    kwargs = _expert().__dict__.copy()
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        ExpertChunk(**kwargs)


def test_store_lock_schema_and_explicit_partial_recovery(tmp_path):
    store = ChunkStore(tmp_path)
    store.append(_expert())
    record = next(store.records())
    assert record["schema_version"] == 2
    store.lock_path.write_text("pid=999999\n")
    with pytest.raises(RuntimeError, match="active writer"):
        store.append(_expert("expert-2"))
    store.lock_path.unlink()
    with store.manifest_path.open("ab") as stream:
        stream.write(b'{"partial"')
    assert store.audit()["manifest"]
    recovered = store.recover()
    assert recovered["truncated_manifest_bytes"] > 0
    assert len(store) == 1


def test_store_detects_orphan_and_invalid_relative_path(tmp_path):
    store = ChunkStore(tmp_path)
    store.append(_expert())
    np.savez(store.payload_dir / "orphan.npz", x=np.zeros(1))
    assert len(store.audit()["orphan_payloads"]) == 1
    record = next(store.records())
    record["payload"] = "../escape.npz"
    store.manifest_path.write_text(json.dumps(record) + "\n")
    with pytest.raises(StorageIntegrityError, match="invalid relative"):
        list(store.records())


def test_teacher_cache_key_changes_for_every_sampling_identity_field():
    base = {
        "observation_id": "obs",
        "teacher_checkpoint": "teacher",
        "teacher_revision": "revision",
        "processor_revision": "processor",
        "inference_steps": 10,
        "sampling_seed": 7,
        "sample_index": 0,
    }
    baseline = TeacherQuery.make_cache_key(**base)
    alternatives = []
    replacements = {
        "observation_id": "obs-2",
        "teacher_checkpoint": "teacher-2",
        "teacher_revision": "revision-2",
        "processor_revision": "processor-2",
        "inference_steps": 11,
        "sampling_seed": 8,
        "sample_index": 1,
    }
    for name, value in replacements.items():
        changed = dict(base)
        changed[name] = value
        alternatives.append(TeacherQuery.make_cache_key(**changed))
    assert baseline not in alternatives
    assert len(set(alternatives)) == len(alternatives)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, value):
        return self.linear(value)


def _metadata() -> dict[str, object]:
    return {
        "base_checkpoint": "student",
        "base_revision": "student-revision",
        "teacher_checkpoint": "teacher",
        "teacher_revision": "teacher-revision",
        "lerobot_commit": "commit",
        "dataset_revision": "dataset-revision",
        "processor_metadata": {"student": "processor"},
        "training_round": 0,
        "policy_version": "B2-step2",
        "replay_manifest_cursor": 0,
        "resolved_config": {"tmd": {"inner_source_mode": "gaussian_tm"}},
        "outer_steps": 2,
        "inner_steps": 2,
        "inner_source_mode": "gaussian_tm",
        "architecture": {"input": 3, "output": 2},
    }


def _training_step(model, discriminator, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    features = torch.randn(4, 3)
    target = torch.randn(4, 2)
    loss = (model(features) - target).square().mean() + 0.01 * discriminator.weight.square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    return float(loss.detach())


def test_interrupted_resume_matches_uninterrupted_training(tmp_path):
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    uninterrupted = TinyPolicy()
    discriminator = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(
        [*uninterrupted.parameters(), *discriminator.parameters()], lr=1e-2
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)
    _training_step(uninterrupted, discriminator, optimizer, scheduler)
    checkpoint = tmp_path / "resume.pt"
    save_training_checkpoint(
        checkpoint,
        policy=uninterrupted,
        discriminator=discriminator,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        metadata=_metadata(),
    )
    expected_loss = _training_step(uninterrupted, discriminator, optimizer, scheduler)
    expected_policy = {name: value.clone() for name, value in uninterrupted.state_dict().items()}
    expected_discriminator = {
        name: value.clone() for name, value in discriminator.state_dict().items()
    }
    expected_random = (random.random(), float(np.random.rand()), float(torch.rand(())))

    resumed = TinyPolicy()
    resumed_discriminator = nn.Linear(2, 1, bias=False)
    resumed_optimizer = torch.optim.AdamW(
        [*resumed.parameters(), *resumed_discriminator.parameters()], lr=1e-2
    )
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, 1, gamma=0.9)
    metadata = load_training_checkpoint(
        checkpoint,
        policy=resumed,
        discriminator=resumed_discriminator,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=None,
    )
    resumed_loss = _training_step(
        resumed, resumed_discriminator, resumed_optimizer, resumed_scheduler
    )
    assert resumed_loss == expected_loss
    for name, value in resumed.state_dict().items():
        torch.testing.assert_close(value, expected_policy[name], rtol=0, atol=0)
    for name, value in resumed_discriminator.state_dict().items():
        torch.testing.assert_close(value, expected_discriminator[name], rtol=0, atol=0)
    assert (random.random(), float(np.random.rand()), float(torch.rand(()))) == expected_random
    assert metadata["inner_source_mode"] == "gaussian_tm"


def test_path_normalizer_refuses_held_out_fit_and_uses_train_statistics():
    normalizer = PathNormalizer(2, 1)
    states = torch.tensor([[[0.0, 2.0], [2.0, 4.0]]])
    actions = torch.tensor([[[3.0]]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="training split"):
        normalizer.fit(states, actions, valid, split="test")
    normalizer.fit(states, actions, valid, split="train")
    normalized_states, normalized_actions = normalizer(states, actions)
    torch.testing.assert_close(normalized_states.mean(dim=(0, 1)), torch.zeros(2))
    torch.testing.assert_close(normalized_actions, torch.zeros_like(normalized_actions))


def test_per_sample_weights_change_only_selected_distillation_contribution():
    expert = torch.tensor([1.0, 1.0])
    teacher = torch.tensor([2.0, 8.0])
    first = combined_distillation_loss(expert, teacher, torch.tensor([1.0, 1.0]))
    second = combined_distillation_loss(expert, teacher, torch.tensor([2.0, 1.0]))
    assert second < first


def test_bootstrap_resamples_complete_episode_rows_deterministically():
    values = np.array([0.0, 1.0, 0.0, 1.0])
    first = bootstrap_episode_statistic(values, task_ids=[0, 0, 1, 1], resamples=100, seed=3)
    second = bootstrap_episode_statistic(values, task_ids=[0, 0, 1, 1], resamples=100, seed=3)
    assert first == second
    assert first["episodes"] == 4


class SeededFakePolicy:
    config = type("Config", (), {"chunk_size": 50, "max_action_dim": 7})()
    generator = type("Generator", (), {"outer_steps": 2})()

    def predict_action_chunk(self, batch, noise=None, *, inner_noises=None):
        del batch
        return noise + 0.01 * inner_noises.sum(dim=0)


def test_action_generation_replays_from_outer_and_inner_seeds():
    runner = CanonicalChunkRunner(SeededFakePolicy(), lambda value: value, lambda value: value)
    observation = {"observation.state": torch.zeros(1, 8)}
    first = runner.plan(observation, "task", 7, (11, 13))
    second = runner.plan(observation, "task", 7, (11, 13))
    changed = runner.plan(observation, "task", 7, (11, 17))
    assert np.array_equal(first.actions, second.actions)
    assert not np.array_equal(first.actions, changed.actions)
    assert first.outer_noise_seed == 7
    assert first.inner_noise_seeds == (11, 13)
    assert min(
        first.preprocessing_latency_s, first.model_latency_s, first.postprocessing_latency_s
    ) >= 0


class NeverTruncatingEnv:
    def __init__(self):
        self.step_count = 0

    def observation(self):
        return {"observation.state": torch.full((1, 8), float(self.step_count))}

    def reset(self, seed):
        del seed
        self.step_count = 0
        return self.observation(), {}

    def step(self, action):
        del action
        self.step_count += 1
        return self.observation(), np.array([0.0]), np.array([False]), np.array([False]), {}


class ZeroPlanRunner:
    def plan(self, observation, instruction, outer_noise_seed):
        del observation, instruction
        return PlanResult(
            actions=np.zeros((50, 7), dtype=np.float32),
            outer_noise_seed=outer_noise_seed,
            inner_noise_seeds=(31, 37),
            preprocessing_latency_s=0.0,
            model_latency_s=0.0,
            postprocessing_latency_s=0.0,
        )


def test_collector_enforces_local_time_limit_when_dependency_never_truncates(monkeypatch):
    monkeypatch.setattr(
        "tmd_policy.rollout.collector._canonical_observation", lambda raw, processor: raw
    )
    records, metrics = collect_rollout_episode(
        NeverTruncatingEnv(),
        None,
        ZeroPlanRunner(),
        policy_checkpoint="student",
        policy_version="B0-test",
        collection_round=0,
        task_index=0,
        instruction="task",
        reset_seed=7,
        base_noise_seed=17,
        max_environment_steps=3,
    )
    assert metrics["environment_steps"] == 3
    assert metrics["local_time_limit_reached"] == 1
    assert records[-1].truncated
    assert records[-1].executed_actions.shape == (3, 7)
    assert records[-1].path_states.shape == (4, 8)
