from __future__ import annotations

from pathlib import Path

import torch

from tmd_policy.evaluation.libero_plus import (
    CATEGORY_NAMES,
    TOTAL_TASKS,
    select_category_sample,
    summarize_libero_plus,
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


def test_libero_plus_config_and_cli_are_wired() -> None:
    from tmd_policy.cli import build_parser
    from tmd_policy.config import load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs/evaluation/libero_plus_dmd2.yaml",
        expected_method="evaluate_libero_plus",
    )
    assert config["evaluation"]["expected_total_tasks"] == 10_030
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
