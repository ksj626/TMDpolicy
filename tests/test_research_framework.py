from __future__ import annotations

import argparse
import inspect
import math
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tmd_policy.common.capabilities import Capability, CapabilitySet
from tmd_policy.common.checkpointing import load_method_checkpoint, save_method_checkpoint
from tmd_policy.common.data import ResearchRecord, ResearchStore, assert_episode_disjoint
from tmd_policy.common.density import RectifiedFlowSchedule, cnf_log_density
from tmd_policy.common.evaluation import (
    EpisodeEvaluation,
    EpisodeOutcome,
    average_precision,
    precision_recall_auc,
    summarize_policy_evaluation,
    wilson_interval,
)
from tmd_policy.common.provenance import capture_git
from tmd_policy.common.tasks import TaskIdentity, TaskRegistry
from tmd_policy.common.teacher import TeacherCacheIdentity
from tmd_policy.methods.base import ResearchMethod
from tmd_policy.methods.dmd2_flow import (
    ConditionalActionGAN,
    DMD2Config,
    DMD2FlowMethod,
    dmd2_distribution_matching_loss,
)
from tmd_policy.methods.flow_sft import FlowSFTConfig, FlowSFTMethod, flow_sft_loss
from tmd_policy.methods.occupancy_tmd import (
    OccupancyDiscriminatorMethod,
    OccupancyGate,
    OccupancyTMDConfig,
    OccupancyTMDMethod,
    OccupancyWindowDiscriminator,
)
from tmd_policy.methods.opd_on_policy import (
    OPDConfig,
    OPDMethod,
    Pi05ProbabilityCapability,
    categorical_opd_loss,
    continuous_flow_opd_loss,
)
from tmd_policy.methods.tmd import (
    ActionMeanFlowHead,
    MeanFlowConfig,
    TMDMethod,
    inner_flow_rollout,
    meanflow_loss,
    meanflow_total_derivative,
)
from tmd_policy.models.smolvla_tmd import SmolVLATMDPolicy
from tmd_policy.research_cli import build_report
from tmd_policy.training.checkpoint import load_policy_for_inference, save_training_checkpoint


def _task(episode: int = 0, instruction: str = "put mug on plate") -> TaskIdentity:
    return TaskIdentity.create(
        benchmark="LIBERO", suite="libero_10", suite_task_index=0,
        dataset_task_index=4, dataset_episode_index=episode, instruction=instruction,
        bddl_identifier="libero_10/task.bddl", bddl_file_hash="a" * 64,
        environment_version="test", source_dataset_id="dataset", source_dataset_revision="b" * 40,
    )


def test_task_registry_allows_many_episodes_but_rejects_cross_task_join() -> None:
    registry = TaskRegistry((_task(0), _task(1)))
    assert registry.by_uid(_task(0).canonical_task_uid).suite_task_index == 0
    other = _task(2, "open drawer")
    with pytest.raises(ValueError, match="canonical task mismatch"):
        registry.assert_joinable((_task(0), other))


def test_research_store_hashes_payload_and_detects_tampering(tmp_path) -> None:
    record = ResearchRecord(
        "expert", "sample-0", _task(), 0, 0, "train",
        {"actions": np.zeros((2, 7), np.float32), "image::front": np.zeros((3, 8, 8), np.uint8)},
    )
    store = ResearchStore(tmp_path)
    path = store.append(record)
    row = next(store.records())
    assert row["content_hash"] == record.content_hash and len(row["payload_sha256"]) == 64
    assert store.audit()["issues"] == []
    path.write_bytes(path.read_bytes() + b"corrupt")
    assert store.audit()["issues"]


def test_images_and_all_invalid_masks_fail_closed() -> None:
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        ResearchRecord("expert", "x", _task(), 0, 0, "train", {
            "image::front": np.full((3, 2, 2), 2.0, np.float32)
        })
    actions = torch.zeros(1, 2, 1)
    with pytest.raises(ValueError, match="valid action"):
        flow_sft_loss(lambda x, t: x, actions, actions, torch.zeros(1), torch.zeros(1, 2, dtype=torch.bool))


def test_flow_sft_equation_and_sampling_sign() -> None:
    action = torch.tensor([[[2.0]]])
    noise = torch.tensor([[[5.0]]])
    time = torch.tensor([0.25])
    seen = {}
    loss = flow_sft_loss(
        lambda state, _: seen.setdefault("state", state) * 0 + 3,
        action, noise, time, torch.ones(1, 1, dtype=torch.bool),
    )
    assert torch.allclose(seen["state"], torch.tensor([[[2.75]]])) and torch.equal(loss, torch.zeros(1))
    # Euler from noise at t=1 toward data at t=0 uses negative dt.
    assert torch.allclose(noise - (noise - action), action)


def test_rectified_velocity_to_score_on_analytic_gaussian() -> None:
    schedule = RectifiedFlowSchedule(0.01, 0.99)
    state, time, data_variance = torch.tensor([[[1.7]]]), torch.tensor([0.4]), 2.0
    variance = (1 - time) ** 2 * data_variance + time**2
    velocity_coefficient = (time - (1 - time) * data_variance) / variance
    score = schedule.velocity_to_score(state, velocity_coefficient[:, None, None] * state, time)
    assert torch.allclose(score, -state / variance[:, None, None], atol=1e-6)
    with pytest.raises(ValueError, match="support"):
        schedule.velocity_to_score(state, state, torch.tensor([0.0]))


def test_exact_cnf_density_and_continuous_opd_action_stop_gradient() -> None:
    action = torch.tensor([[[0.5]]], requires_grad=True)
    zero = lambda state, time: state * 0
    expected = -0.5 * (0.25 + math.log(2 * math.pi))
    assert torch.allclose(cnf_log_density(action, zero, steps=2), torch.tensor([expected]))
    translation = lambda state, time: state * 0 + 0.25
    translated_expected = -0.5 * (0.75**2 + math.log(2 * math.pi))
    assert torch.allclose(
        cnf_log_density(action, translation, steps=2), torch.tensor([translated_expected])
    )
    scale = nn.Parameter(torch.tensor(0.1))
    result = continuous_flow_opd_loss(
        action,
        student_vector_field=lambda state, time: scale * state,
        teacher_vector_field=zero,
        integration_steps=2,
        divergence_mode="exact",
    )
    result["loss"].backward()
    assert scale.grad is not None and action.grad is None


def test_categorical_opd_reward_is_detached_and_direction_is_reverse_kl() -> None:
    student = torch.tensor([[[2.0, 0.0]]], requires_grad=True)
    teacher = torch.tensor([[[0.0, 2.0]]], requires_grad=True)
    result = categorical_opd_loss(student, teacher, torch.zeros((1, 1), dtype=torch.long), torch.ones(1, 1, dtype=torch.bool))
    assert not result["reward"].requires_grad and result["reward"].item() < 0
    result["loss"].backward()
    assert student.grad is not None and teacher.grad is None


def test_meanflow_jvp_matches_finite_difference_and_inner_sign() -> None:
    torch.manual_seed(0)
    config = MeanFlowConfig(action_dim=1, feature_dim=2, hidden_dim=4)
    head = ActionMeanFlowHead(config).double()
    assert "inner_source" not in inspect.signature(head.forward).parameters
    y = torch.randn(2, 3, 1, dtype=torch.double)
    features = torch.randn(2, 3, 2, dtype=torch.double)
    s, r = torch.tensor([0.4, 0.7], dtype=torch.double), torch.tensor([0.1, 0.2], dtype=torch.double)
    source, velocity = torch.randn_like(y), torch.randn_like(y)
    jvp = meanflow_total_derivative(head, y, s, r, features, source, velocity, mode="jvp", delta=1e-4)
    fd = meanflow_total_derivative(head, y, s, r, features, source, velocity, mode="finite_difference", delta=1e-4)
    assert torch.allclose(jvp, fd, atol=2e-4, rtol=2e-3)

    class Oracle(nn.Module):
        def forward(self, state, current, target, feature):
            return torch.zeros_like(state)

    initial = torch.ones(1, 2, 1)
    final = inner_flow_rollout(Oracle(), inner_source=initial, features=torch.zeros(1, 2, 2), time_grid=torch.tensor([1.0, 0.0]))
    assert torch.equal(final, torch.zeros_like(final))

    float_config = MeanFlowConfig(action_dim=1, feature_dim=2, hidden_dim=4)
    float_head = ActionMeanFlowHead(float_config)
    result = meanflow_loss(
        float_head,
        outer_data=torch.zeros(1, 2, 1),
        outer_source=torch.ones(1, 2, 1),
        outer_time=torch.tensor([0.5]),
        inner_source=torch.randn(1, 2, 1),
        inner_time=torch.tensor([0.6]),
        target_time=torch.tensor([0.2]),
        features=torch.randn(1, 2, 2),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        config=float_config,
    )
    assert not result["target"].requires_grad
    result["loss"].backward()
    assert any(parameter.grad is not None for parameter in float_head.parameters())


def test_tmd_backbone_head_counts_and_deterministic_inner_noise_replay() -> None:
    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(1.0))
            self.calls = 0

        def forward(self, state, time):
            self.calls += 1
            return torch.zeros(*state.shape[:2], 2) * self.scale

    backbone = Backbone()
    head = ActionMeanFlowHead(MeanFlowConfig(action_dim=1, feature_dim=2, hidden_dim=4))
    head_calls = []
    hook = head.register_forward_hook(lambda *args: head_calls.append(1))
    method = TMDMethod(
        backbone=backbone, head=head, config=head.config, stage=1,
        capabilities=CapabilitySet.of("test", Capability.EXPERT_ACTION_CHUNKS, Capability.FLOW_VELOCITY),
    )
    batch = {
        "outer_state": torch.ones(1, 2, 1), "outer_time": torch.tensor([0.5]),
        "inner_source": torch.randn(1, 2, 1), "inner_time_grid": torch.tensor([1.0, 0.5, 0.0]),
        "outer_step": torch.tensor(0.5),
    }
    first, second = method.sample_action_chunk(batch), method.sample_action_chunk(batch)
    hook.remove()
    assert torch.equal(first, second)
    assert backbone.calls == 2 and len(head_calls) == 4


def test_dmd2_generator_gradient_is_detached_from_score_parameters() -> None:
    generated = torch.ones(1, 2, 1, requires_grad=True)
    real_scale, fake_scale = nn.Parameter(torch.tensor(0.2)), nn.Parameter(torch.tensor(-0.1))
    result = dmd2_distribution_matching_loss(
        generated, time=torch.tensor([0.5]), corruption_noise=torch.zeros_like(generated),
        valid_mask=torch.ones(1, 2, dtype=torch.bool),
        real_velocity=lambda state, time: real_scale * state,
        fake_velocity=lambda state, time: fake_scale * state,
        schedule=RectifiedFlowSchedule(),
    )
    result["loss"].backward()
    assert generated.grad is not None and real_scale.grad is None and fake_scale.grad is None


def test_occupancy_discriminator_is_causal() -> None:
    torch.manual_seed(2)
    model = OccupancyWindowDiscriminator(state_dim=2, action_dim=1, task_count=2, model_dim=8, layers=1, heads=2).eval()
    states, actions = torch.randn(1, 4, 2), torch.randn(1, 3, 1)
    mask, tasks = torch.ones(1, 3, dtype=torch.bool), torch.zeros(1, dtype=torch.long)
    original = model(states, actions, tasks, mask)
    actions[:, 2] += 100
    changed = model(states, actions, tasks, mask)
    assert torch.allclose(original[:, :2], changed[:, :2], atol=1e-6)


def test_episode_splits_metrics_and_dirty_git_provenance(tmp_path) -> None:
    with pytest.raises(ValueError, match="disjointness"):
        assert_episode_disjoint(((1, 0, "train"), (1, 10, "test")))
    labels, scores = [1, 0, 1], [0.9, 0.8, 0.1]
    assert average_precision(labels, scores) != precision_recall_auc(labels, scores)
    low, high = wilson_interval(0, 3)
    assert low == 0 and high > 0
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=tmp_path, check=True)
    (tmp_path / "a").write_text("one")
    subprocess.run(["git", "add", "a"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "one"], cwd=tmp_path, check=True)
    (tmp_path / "a").write_text("two")
    (tmp_path / "new").write_text("untracked")
    provenance = capture_git(tmp_path)
    assert provenance.dirty and "untracked" in provenance.patch and len(provenance.patch_sha256) == 64


def test_shared_evaluator_reports_nonzero_uncertainty_for_three_failures() -> None:
    episodes = [
        EpisodeEvaluation(
            "task", seed, EpisodeOutcome(False, False, False, True, 2), np.zeros((2, 1), np.float32),
            (0.01,), (0.02,), (0.01,), (0.03,), 1.0, None,
        )
        for seed in range(3)
    ]
    report = summarize_policy_evaluation(episodes)
    assert report["micro_success"] == 0
    assert report["micro_wilson_95"][1] > 0
    assert "memory_protocol" in report and "action_smoothness_mean_l2_delta" in report


def test_every_research_config_propagates_to_dry_run() -> None:
    import yaml

    config_root = __import__("pathlib").Path(__file__).parents[1] / "configs"
    research_paths = [
        path
        for folder in ("methods", "data", "evaluation", "experiments")
        for path in (config_root / folder).glob("*.yaml")
        if path.name not in {"task_registry.json"}
    ]
    for path in sorted(research_paths):
        args = argparse.Namespace(config=str(path), operation="test", execute=False, resume=None)
        report = build_report(args)
        assert report["resolved_config"] == yaml.safe_load(path.read_text())
        assert report["resolved_config"]["method"] == report["method"]
        assert report["dataset_selection"]["revision"]
        assert report["task_mappings"]


def test_legacy_tmd_loss_and_dropout_reach_instantiated_head() -> None:
    base = nn.Module()
    base.config = SimpleNamespace(max_action_dim=7, chunk_size=50)
    base.model = SimpleNamespace(vlm_with_expert=SimpleNamespace(expert_hidden_size=8))
    policy = SmolVLATMDPolicy(
        base, hidden_dim=4, recurrent_layers=1, dropout=0.37, transition_loss="mse"
    )
    assert policy.generator.transition_head.dropout.p == 0.37
    assert policy.generator.transition_loss == "mse"


def test_teacher_cache_key_resists_action_and_schedule_collisions() -> None:
    base = TeacherCacheIdentity("a" * 64, "task", "b" * 40, "c" * 40, (1.0, 0.0), 7, 0, "d" * 64)
    changed = TeacherCacheIdentity("a" * 64, "task", "b" * 40, "c" * 40, (1.0, 0.5, 0.0), 7, 0, "d" * 64)
    changed_action = TeacherCacheIdentity("a" * 64, "task", "b" * 40, "c" * 40, (1.0, 0.0), 7, 0, "e" * 64)
    assert len({base.key, changed.key, changed_action.key}) == 3


def test_format_v3_checkpoint_restores_model_optimizer_scheduler_and_rng(tmp_path) -> None:
    torch.manual_seed(4)
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1)
    loss = model(torch.ones(1, 2)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    path = save_method_checkpoint(
        tmp_path / "state.pt", method_name="tiny", models={"model": model},
        optimizers={"optimizer": optimizer}, schedulers={"scheduler": scheduler}, scaler=None,
        counters={"step": 1}, config={"x": 2}, provenance={"commit": "abc"},
        trainable_names={"model": [name for name, _ in model.named_parameters()]},
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    value = load_method_checkpoint(
        path, expected_method="tiny", models={"model": model}, optimizers={"optimizer": optimizer},
        schedulers={"scheduler": scheduler},
    )
    assert value["counters"] == {"step": 1} and value["provenance"] == {"commit": "abc"}
    assert all(torch.equal(model.state_dict()[key], tensor) for key, tensor in expected.items())


def test_action_projection_trainability_and_values_survive_inference_checkpoint(tmp_path) -> None:
    class Policy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.generator = nn.Module()
            self.generator.transition_head = nn.Linear(2, 2)
            self.action_projection = nn.Linear(2, 2)

        def configure(self, projections: bool) -> None:
            self.requires_grad_(False)
            self.generator.transition_head.requires_grad_(True)
            self.action_projection.requires_grad_(projections)

    policy = Policy()
    policy.configure(True)
    with torch.no_grad():
        policy.action_projection.weight.fill_(3.0)
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=0.01)
    metadata = {
        "base_checkpoint": "student", "base_revision": "a" * 40,
        "teacher_checkpoint": "teacher", "teacher_revision": "b" * 40,
        "lerobot_commit": "c" * 40, "dataset_revision": "d" * 40,
        "processor_metadata": {}, "training_round": 0, "policy_version": "v1",
        "replay_manifest_cursor": 0, "resolved_config": {}, "outer_steps": 2,
        "inner_steps": 2, "inner_source_mode": "gaussian_tm", "architecture": {},
        "train_main_action_projections": True,
    }
    path = save_training_checkpoint(
        tmp_path / "policy.pt", policy=policy, discriminator=None, optimizer=optimizer,
        scheduler=None, scaler=None, metadata=metadata,
    )
    restored = Policy()
    restored.configure(True)
    loaded = load_policy_for_inference(path, restored)
    assert loaded["train_main_action_projections"] is True
    assert torch.equal(restored.action_projection.weight, policy.action_projection.weight)


def test_dmd2_ttur_update_order_and_parameter_isolation_without_optimizer_steps() -> None:
    events: list[tuple[str, tuple[bool, bool, bool]]] = []

    class Field(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(1, 1)

        def forward(self, state, time, condition):
            return self.linear(state) + condition[:, None, :1] * 0

    generator, fake = Field(), Field()
    discriminator = ConditionalActionGAN(1, 1, hidden_dim=4)
    method = DMD2FlowMethod(
        generator=generator,
        fake_score=fake,
        discriminator=discriminator,
        teacher_velocity=lambda state, time, condition: state * 0,
        config=DMD2Config(fake_updates_per_generator=5),
        capabilities=CapabilitySet.of(
            "test", Capability.EXPERT_ACTION_CHUNKS, Capability.FLOW_SCORE,
            Capability.TEACHER_AT_STUDENT_ACTION,
        ),
    )

    class NoStep:
        def __init__(self, name, parameters):
            self.name, self.parameters = name, tuple(parameters)

        def zero_grad(self, set_to_none=True):
            for parameter in self.parameters:
                parameter.grad = None

        def step(self):
            events.append((self.name, (
                next(generator.parameters()).requires_grad,
                next(fake.parameters()).requires_grad,
                next(discriminator.parameters()).requires_grad,
            )))

    class NoSchedule:
        def step(self):
            pass

    method.optimizers = {
        "generator": NoStep("generator", generator.parameters()),
        "fake_score": NoStep("fake_score", fake.parameters()),
        "discriminator": NoStep("discriminator", discriminator.parameters()),
    }
    method.schedulers = {name: NoSchedule() for name in method.optimizers}
    one = torch.ones(1, 2, 1)
    method.training_step({
        "condition": torch.zeros(1, 1), "valid_mask": torch.ones(1, 2, dtype=torch.bool),
        "condition_task_uids": ("task",), "gan_real_task_uids": ("task",),
        "simulation_initial_noise": one, "simulation_reinjection_noises": (one, one, one),
        "simulation_step_index": 0, "fake_noise": torch.zeros_like(one),
        "fake_time": torch.tensor([0.5]), "gan_real_noisy": torch.zeros_like(one),
        "score_time": torch.tensor([0.5]), "score_noise": torch.zeros_like(one),
        "gan_generator_noise": torch.zeros_like(one), "gan_time": torch.tensor([0.5]),
    })
    assert [name for name, _ in events] == ["fake_score", "discriminator"] * 5 + ["generator"]
    assert all(flags == (False, True, True) for _, flags in events[:-1])
    assert events[-1] == ("generator", (True, False, False))


def test_one_batch_method_interfaces_and_pi05_probability_gate(tmp_path) -> None:
    class FlowPolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.action_layer = nn.Linear(1, 1)

        def forward(self, batch, reduction="none"):
            loss = self.action_layer(batch["action"]).square().mean((1, 2))
            return loss, {}

        def predict_action_chunk(self, batch):
            return self.action_layer(batch["action"])

    flow = FlowSFTMethod(policy=FlowPolicy(), config=FlowSFTConfig(mixed_precision="none"))
    flow_result = flow.training_step({
        "action": torch.ones(2, 3, 1), "action_is_pad": torch.zeros(2, 3, dtype=torch.bool)
    })
    assert flow_result["per_sample_loss"].shape == (2,)

    student, teacher = nn.Linear(1, 1), nn.Linear(1, 1)
    opd = OPDMethod(
        mode="opd_categorical", student=student, teacher=teacher,
        config=OPDConfig(group_size=2),
        capabilities=CapabilitySet.of("tokens", Capability.ON_POLICY_ROLLOUTS, Capability.TOKEN_LOG_PROBABILITY),
        current_policy_version="v0",
    )
    opd_result = opd.training_step({
        "policy_versions": ("v0", "v0"), "collection_rounds": (0, 0), "group_ids": (1, 1),
        "student_logits": torch.randn(2, 1, 3, requires_grad=True),
        "teacher_logits": torch.randn(2, 1, 3), "sampled_tokens": torch.zeros(2, 1, dtype=torch.long),
        "valid_mask": torch.ones(2, 1, dtype=torch.bool),
    })
    assert opd_result["per_trajectory_loss"].shape == (2,)
    pi05 = Pi05ProbabilityCapability("a" * 40, "b" * 40)
    with pytest.raises(RuntimeError, match="unavailable capabilities"):
        pi05.capability_set().require({Capability.EXACT_LOG_DENSITY}, method="continuous_flow_opd")

    discriminator = OccupancyWindowDiscriminator(
        state_dim=2, action_dim=1, task_count=1, model_dim=4, layers=1, heads=2
    )
    disc_method = OccupancyDiscriminatorMethod(
        discriminator=discriminator, config=OccupancyTMDConfig(),
        capabilities=CapabilitySet.of("paths", Capability.PATH_WINDOWS, Capability.ON_POLICY_ROLLOUTS),
    )
    paths = {
        "expert_states": torch.randn(2, 3, 2), "expert_actions": torch.randn(2, 2, 1),
        "expert_task_ids": torch.zeros(2, dtype=torch.long),
        "expert_valid_mask": torch.ones(2, 2, dtype=torch.bool),
        "student_states": torch.randn(2, 3, 2), "student_actions": torch.randn(2, 2, 1),
        "student_task_ids": torch.zeros(2, dtype=torch.long),
        "student_valid_mask": torch.ones(2, 2, dtype=torch.bool),
    }
    assert disc_method.training_step(paths)["loss"].ndim == 0

    class Downstream(ResearchMethod):
        name, classification = "downstream", "test"
        def required_data_capabilities(self): return frozenset()
        def validate_config(self): return None
        def training_step(self, batch): return {"loss": batch["prioritization_weight"].mean()}
        def sample_action_chunk(self, batch): return batch["actions"]
        def save_method_state(self, path): return tmp_path
        def load_method_state(self, path): return {}
        def dry_run_report(self): raise RuntimeError

    weighted = OccupancyTMDMethod(
        tmd_stage2=Downstream(), discriminator=discriminator, config=OccupancyTMDConfig(),
        gate=OccupancyGate(True, 0.0, 0.0, 0.0, 1.0, 20.0),
        capabilities=CapabilitySet.of(
            "all", Capability.PATH_WINDOWS, Capability.ON_POLICY_ROLLOUTS, Capability.FLOW_SCORE
        ),
    )
    weighted_result = weighted.training_step({
        "states": paths["student_states"], "actions": paths["student_actions"],
        "task_ids": paths["student_task_ids"], "valid_mask": paths["student_valid_mask"],
    })
    assert weighted_result["loss"].ndim == 0
