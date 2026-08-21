from __future__ import annotations

from pathlib import Path

import torch

from tmd_policy.evaluation.libero_plus import (
    CATEGORY_NAMES,
    TOTAL_TASKS,
    _difficulty_level,
    resolve_video_setting,
    select_category_sample,
    summarize_libero_plus,
)
from tmd_policy.evaluation.policy import _dmd2_seeded_noises, _seeded_noise
from tmd_policy.libero_protocol import (
    LIBERO_SUITE_MAX_EPISODE_STEPS,
    init_state_index_for_trial,
    set_fixed_init_state_index,
)
from tmd_policy.training.rollout_replay import AsyncStudentRolloutManager, BalancedStudentReplay


def _record(task: int, value: int) -> dict:
    return {
        "state": torch.full((8,), float(value)),
        "instruction": f"task {task}",
        "global_task_index": task,
        "observations": {
            "observation.images.image": torch.full((1, 3, 4, 4), value, dtype=torch.uint8),
            "observation.images.image2": torch.full((1, 3, 4, 4), value, dtype=torch.uint8),
        },
    }


def test_balanced_student_replay_round_robins_across_task_queues() -> None:
    replay = BalancedStudentReplay(80, seed=3)
    replay._records[0].append(_record(0, 1))
    replay._records[17].append(_record(17, 2))
    batch = replay.sample_like({"observation.state": torch.zeros(4, 8)})
    assert batch is not None
    assert batch["task_index"].tolist() == [0, 17, 0, 17]
    assert batch["observation.images.image"].shape == (4, 3, 4, 4)
    assert batch["action_is_pad"].shape == (4, 50)
    assert bool(batch["_student_replay_batch"])


def test_libero_protocol_uses_suite_horizons_and_explicit_fixed_init_states() -> None:
    assert LIBERO_SUITE_MAX_EPISODE_STEPS == {
        "libero_spatial": 280,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }
    assert init_state_index_for_trial(0) == 0
    assert init_state_index_for_trial(49) == 49
    assert init_state_index_for_trial(50) == 0

    class VectorEnv:
        def __init__(self) -> None:
            self.values = []

        def set_attr(self, name, value):
            self.values.append((name, value))

    env = VectorEnv()
    assert set_fixed_init_state_index(env, 51) == 1
    assert env.values == [("init_state_id", 1)]


def test_libero_plus_summary_covers_10030_contract() -> None:
    assert TOTAL_TASKS == 10_030
    rows = [
        {
            "suite": "libero_spatial",
            "category": "Camera Viewpoints",
            "difficulty_level": 2,
            "success": True,
            "model_latency_s": [0.1],
            "mean_model_latency_s": 0.1,
            "mean_environment_latency_s": 0.01,
            "mean_action_l2": 1.0,
            "mean_action_delta_l2": 0.2,
        },
        {
            "suite": "libero_object",
            "category": "Sensor Noise",
            "difficulty_level": 4,
            "success": False,
            "model_latency_s": [0.2],
            "mean_model_latency_s": 0.2,
            "mean_environment_latency_s": 0.02,
            "mean_action_l2": 2.0,
            "mean_action_delta_l2": 0.4,
        },
    ]
    summary = summarize_libero_plus(rows, full_benchmark=False)
    assert summary["episodes"] == 2
    assert summary["micro_success_rate"] == 0.5
    assert set(summary["per_category"]) == {"Camera Viewpoints", "Sensor Noise"}


def test_libero_plus_summary_accepts_unassigned_upstream_difficulty() -> None:
    assert _difficulty_level(None) is None
    assert _difficulty_level(3) == 3
    row = {
        "suite": "libero_goal",
        "category": "Light Conditions",
        "difficulty_level": None,
        "success": False,
        "model_latency_s": [0.1],
        "mean_model_latency_s": 0.1,
        "mean_environment_latency_s": 0.01,
        "mean_action_l2": 1.0,
        "mean_action_delta_l2": 0.2,
    }
    summary = summarize_libero_plus([row], full_benchmark=False)
    assert summary["per_difficulty"]["None"]["episodes"] == 1


def test_libero_plus_config_and_cli_are_wired() -> None:
    from tmd_policy.cli import build_parser
    from tmd_policy.config import load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs/evaluation/libero_plus_dmd2.yaml",
        expected_method="evaluate_libero_plus",
    )
    assert config["evaluation"]["expected_total_tasks"] == 10_030
    assert config["evaluation"]["batch_size"] == 1
    args = build_parser().parse_args(
        [
            "evaluate",
            "libero-plus",
            "--checkpoint",
            "student.pt",
            "--output",
            "results",
            "--suite",
            "libero_spatial",
            "--task-ids",
            "0",
            "3",
            "--resume",
        ]
    )
    assert args.task_ids == [0, 3]
    assert args.resume

    expected_baselines = {
        "libero_plus_smolvla_official10.yaml": {
            "method": "smolvla",
            "sampler_mode": "official",
        },
        "libero_plus_smolvla_4step_ablation.yaml": {
            "method": "smolvla",
            "sampler_mode": "override",
            "num_steps": 4,
        },
        "libero_plus_smolvla_1step_ablation.yaml": {
            "method": "smolvla",
            "sampler_mode": "override",
            "num_steps": 1,
        },
        "libero_plus_pi05_official.yaml": {
            "method": "pi05",
            "num_steps": 10,
        },
    }
    for filename, expected_policy in expected_baselines.items():
        baseline = load_config(
            root / "configs/evaluation" / filename,
            expected_method="evaluate_libero_plus",
        )
        assert baseline["evaluation"] == config["evaluation"]
        for key, value in expected_policy.items():
            assert baseline["policy"][key] == value

    sampled = build_parser().parse_args(
        [
            "evaluate",
            "libero-plus",
            "--sample-per-category",
            "10",
            "--sample-seed",
            "19",
        ]
    )
    assert sampled.sample_per_category == 10
    assert sampled.sample_seed == 19

    accelerated = build_parser().parse_args(
        [
            "evaluate",
            "libero-plus",
            "--suite",
            "libero_spatial",
            "libero_goal",
            "--suite",
            "libero_10",
            "--batch-size",
            "6",
            "--devices",
            "cuda:2",
            "cuda:3",
            "--no-save-videos",
        ]
    )
    assert accelerated.suite == ["libero_spatial", "libero_goal", "libero_10"]
    assert accelerated.batch_size == 6
    assert accelerated.devices == ["cuda:2", "cuda:3"]
    assert accelerated.save_videos is False

    bootstrap = build_parser().parse_args(
        [
            "train",
            "dmd2-flow",
            "--initial-rollout-replay",
            "/tmp/round-000000",
        ]
    )
    assert bootstrap.initial_rollout_replay == "/tmp/round-000000"


def test_libero_plus_category_sample_has_exact_balanced_counts() -> None:
    from collections import Counter

    suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    classification = {
        suite: [
            {"category": category}
            for category in CATEGORY_NAMES
            for _ in range(12)
        ]
        for suite in suites
    }
    selected = select_category_sample(
        classification, suites, per_category=10, seed=17
    )
    counts = Counter(
        str(classification[suite][task_id]["category"])
        for suite, task_ids in selected.items()
        for task_id in task_ids
    )
    assert counts == Counter({category: 10 for category in CATEGORY_NAMES})
    assert sum(len(task_ids) for task_ids in selected.values()) == 70
    assert selected == select_category_sample(
        classification, suites, per_category=10, seed=17
    )


def test_libero_plus_category_samples_save_videos_by_default() -> None:
    assert resolve_video_setting(sample_per_category=10, save_videos=None)
    assert not resolve_video_setting(sample_per_category=None, save_videos=None)
    assert not resolve_video_setting(sample_per_category=10, save_videos=False)
    assert resolve_video_setting(sample_per_category=None, save_videos=True)


def test_batched_noise_preserves_each_episode_rng_stream() -> None:
    device = torch.device("cpu")
    seeds = [13, 29, 41]
    batched = _seeded_noise(
        seeds,
        batch_size=len(seeds),
        device=device,
        shape=(5, 7),
    )
    for index, seed in enumerate(seeds):
        serial = _seeded_noise(seed, batch_size=1, device=device, shape=(5, 7))[0]
        torch.testing.assert_close(batched[index], serial, rtol=0, atol=0)


def test_batched_dmd2_noise_preserves_initial_and_renoise_streams() -> None:
    device = torch.device("cpu")
    seeds = [101, 303]
    steps = 4
    initial, generator, renoise = _dmd2_seeded_noises(
        seeds,
        batch_size=len(seeds),
        device=device,
        outer_steps=steps,
    )
    assert generator is None
    assert renoise is not None
    assert renoise.shape == (steps - 1, len(seeds), 50, 32)
    for episode_index, seed in enumerate(seeds):
        serial_initial, serial_generator, serial_renoise = _dmd2_seeded_noises(
            seed,
            batch_size=1,
            device=device,
            outer_steps=steps,
        )
        assert serial_generator is not None
        assert serial_renoise is None
        torch.testing.assert_close(
            initial[episode_index], serial_initial[0], rtol=0, atol=0
        )
        for step_index in range(steps - 1):
            expected = torch.randn(
                1,
                50,
                32,
                device=device,
                generator=serial_generator,
            )[0]
            torch.testing.assert_close(
                renoise[step_index, episode_index], expected, rtol=0, atol=0
            )


def test_external_rollout_round_bootstraps_without_collection(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source" / "round-000000"
    source.mkdir(parents=True)
    (source / "collection_report.json").write_text("{}\n", encoding="utf-8")

    def ingest_all_tasks(self, root):
        for task in range(40):
            self._records[task].append(_record(task, task))
        self.rounds.append(str(Path(root).resolve()))
        return 40

    monkeypatch.setattr(BalancedStudentReplay, "ingest", ingest_all_tasks)
    manager = AsyncStudentRolloutManager(
        config={
            "training": {"seed": 7},
            "dmd2": {
                "student_rollout_replay": {
                    "capacity_replans": 80,
                    "refresh_generator_updates": 500,
                }
            },
        },
        output=tmp_path / "new_run",
        initial_round=source,
    )
    assert manager.replay.size == 40
    assert manager.replay.task_support == list(range(40))
    assert not manager.should_refresh(499)
    assert manager.should_refresh(500)
    assert manager.metrics()["replay/bootstrap_source"] == str(source.resolve())


def test_rollout_manager_propagates_serial_multi_gpu_workers(tmp_path) -> None:
    manager = AsyncStudentRolloutManager(
        config={
            "training": {"seed": 7},
            "backend": {},
            "models": {},
            "dataset": {},
            "horizons": {"execution": 10},
            "dmd2": {
                "student_rollout_replay": {
                    "capacity_replans": 80,
                    "device": "cuda:2",
                    "devices": ["cuda:2", "cuda:3"],
                }
            },
        },
        output=tmp_path / "run",
    )
    runtime = manager._rollout_config(tmp_path / "student.pt", tmp_path / "round")
    assert runtime["collection"]["batch_size"] == 1
    assert runtime["collection"]["devices"] == ["cuda:2", "cuda:3"]
    assert runtime["collection"]["train_reset_seeds"] == [100_000]
    assert runtime["collection"]["train_init_state_indices"] == [0]
    assert runtime["collection"]["suite_max_episode_steps"] == (
        LIBERO_SUITE_MAX_EPISODE_STEPS
    )

    manager._round = 51
    later = manager._rollout_config(tmp_path / "student.pt", tmp_path / "later")
    assert later["collection"]["train_reset_seeds"] == [100_051]
    assert later["collection"]["train_init_state_indices"] == [1]


def test_paper_rollout_uses_independent_serial_gpu_shards(tmp_path: Path) -> None:
    from tmd_policy.config import load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/methods/dmd2_flow.yaml")
    replay = config["dmd2"]["student_rollout_replay"]
    assert replay["batch_size"] == 1
    assert replay["devices"] == ["cuda:2", "cuda:3", "cuda:4", "cuda:5"]
    assert replay["suite_max_episode_steps"] == LIBERO_SUITE_MAX_EPISODE_STEPS
    assert replay["validation_videos"] == {
        "enabled": True,
        "task_ids": [0],
        "simulator_seed": 200_000,
        "init_state_index": 49,
        "camera": "image",
    }
    manager = AsyncStudentRolloutManager(
        config=config,
        output=tmp_path / "paper",
    )
    runtime = manager._rollout_config(tmp_path / "student.pt", tmp_path / "round")
    collection = runtime["collection"]
    assert collection["validation_reset_seeds"] == [200_000]
    assert collection["validation_init_state_indices"] == [49]
    assert collection["validation_task_ids"] == [0]
    assert collection["save_validation_videos"]


def test_all_evaluation_configs_use_the_fixed_suite_horizons() -> None:
    from tmd_policy.config import load_config

    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "configs/evaluation").glob("*.yaml")):
        config = load_config(path)
        assert config["evaluation"]["suite_max_episode_steps"] == (
            LIBERO_SUITE_MAX_EPISODE_STEPS
        )
