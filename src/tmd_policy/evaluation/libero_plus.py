"""Resumable one-environment-at-a-time evaluation on all 10,030 LIBERO-Plus tasks."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import io
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from tqdm.auto import tqdm

from tmd_policy.config import save_resolved_config

from .libero import run_episode, wilson_interval
from .policy import load_inference_policy


SUITE_COUNTS = {
    "libero_spatial": 2_402,
    "libero_object": 2_518,
    "libero_goal": 2_591,
    "libero_10": 2_519,
}
SUITE_OFFSETS = {
    "libero_spatial": 0,
    "libero_object": 2_402,
    "libero_goal": 4_920,
    "libero_10": 7_511,
}
TOTAL_TASKS = sum(SUITE_COUNTS.values())
CATEGORY_NAMES = (
    "Background Textures",
    "Camera Viewpoints",
    "Language Instructions",
    "Light Conditions",
    "Objects Layout",
    "Robot Initial States",
    "Sensor Noise",
)


def _atomic_json(value: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def _source_identity(classification_path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(classification_path.read_bytes()).hexdigest()
    repository = classification_path.parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        commit = None
    return {
        "repository": str(repository),
        "git_commit": commit,
        "classification_path": str(classification_path),
        "classification_sha256": digest,
    }


def _comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    result = dict(config)
    result.pop("_config_path", None)
    evaluation = dict(result.get("evaluation", {}))
    evaluation.setdefault("sample_per_category", None)
    evaluation.setdefault("sample_seed", 0)
    result["evaluation"] = evaluation
    return result


def _prepare_output(output: Path, config: dict[str, Any], *, resume: bool) -> None:
    resolved = output / "resolved_config.yaml"
    if output.exists() and not resume:
        raise FileExistsError(
            f"LIBERO-Plus output already exists: {output}; pass --resume to continue it"
        )
    if output.exists() and resume:
        if not resolved.exists():
            raise FileNotFoundError(f"cannot resume without {resolved}")
        previous = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        if _comparable_config(previous) != _comparable_config(config):
            raise ValueError("LIBERO-Plus resume configuration differs from resolved_config.yaml")
    output.mkdir(parents=True, exist_ok=True)
    if not resolved.exists():
        save_resolved_config(config, resolved)


def _load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    identities: set[tuple[str, int, int]] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = (str(row["suite"]), int(row["task_id"]), int(row["reset_seed"]))
            if identity in identities:
                raise ValueError(f"duplicate LIBERO-Plus episode at {path}:{line_number}")
            identities.add(identity)
            rows.append(row)
    return rows


def _load_catalogs(suites: list[str]) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], Path]:
    """Load the fork and fail closed if vanilla LIBERO shadows it."""

    # LIBERO-Plus imports Wand after Robosuite/OpenCV. On mixed system/Conda
    # installations that order can bind an incompatible ImageMagick dependency.
    # Load only MagickWand first; do not put the whole Conda lib directory on
    # LD_LIBRARY_PATH because that shadows the NVIDIA EGL driver with Mesa EGL.
    from wand.api import library as _wand_library  # noqa: F401
    from libero.libero import benchmark

    classification_path = Path(inspect.getfile(benchmark)).resolve().with_name(
        "task_classification.json"
    )
    if not classification_path.exists():
        raise RuntimeError(
            "active `libero` is not LIBERO-Plus (task_classification.json is absent); "
            "run scripts/setup/create_libero_plus_environment.sh and use that environment"
        )
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    mapping = benchmark.get_benchmark_dict()
    catalogs: dict[str, Any] = {}
    for suite in suites:
        # The upstream fork prints a multi-thousand-element task permutation at
        # construction. Keep the durable evaluator log concise.
        with contextlib.redirect_stdout(io.StringIO()):
            catalog = mapping[suite]()
        expected = SUITE_COUNTS[suite]
        metadata = classification.get(suite)
        if len(catalog.tasks) != expected or not isinstance(metadata, list) or len(metadata) != expected:
            raise RuntimeError(
                f"LIBERO-Plus suite {suite} has the wrong size: "
                f"tasks={len(catalog.tasks)}, metadata={len(metadata or [])}, expected={expected}"
            )
        for task_id, (task, row) in enumerate(zip(catalog.tasks, metadata, strict=True)):
            if int(row["id"]) != task_id + 1 or str(row["name"]) != str(task.name):
                raise RuntimeError(
                    f"LIBERO-Plus classification/task ordering mismatch at {suite}:{task_id}"
                )
        catalogs[suite] = catalog
    return catalogs, classification, classification_path


def select_category_sample(
    classification: dict[str, list[dict[str, Any]]],
    suites: list[str],
    *,
    per_category: int,
    seed: int,
) -> dict[str, list[int]]:
    """Select a deterministic category-balanced smoke set across suites."""

    if per_category < 1:
        raise ValueError("LIBERO-Plus sample_per_category must be positive")
    selected: dict[str, list[int]] = {suite: [] for suite in suites}
    for category_index, category in enumerate(CATEGORY_NAMES):
        by_suite: dict[str, list[int]] = {}
        for suite in suites:
            candidates = [
                task_id
                for task_id, row in enumerate(classification[suite])
                if str(row["category"]) == category
            ]
            random.Random(f"{seed}:{category}:{suite}").shuffle(candidates)
            by_suite[suite] = candidates
        available = sum(len(values) for values in by_suite.values())
        if available < per_category:
            raise ValueError(
                f"LIBERO-Plus category {category!r} has only {available} tasks "
                f"across suites {suites}, fewer than requested {per_category}"
            )
        suite_order = list(suites)
        offset = (seed + category_index) % len(suite_order)
        suite_order = suite_order[offset:] + suite_order[:offset]
        chosen = 0
        cursor = 0
        while chosen < per_category:
            suite = suite_order[cursor % len(suite_order)]
            if by_suite[suite]:
                selected[suite].append(by_suite[suite].pop())
                chosen += 1
            cursor += 1
    return {suite: sorted(set(ids)) for suite, ids in selected.items()}


def _single_task_env(
    catalog: Any,
    *,
    suite: str,
    task_id: int,
    fps: int,
    episode_length: int,
    control_mode: str,
    hard_reset: bool,
) -> Any:
    """Build only the current task; never materialize thousands of vector envs."""

    from lerobot.envs.libero import LiberoEnv as LeRobotLiberoEnv

    def factory() -> Any:
        return LeRobotLiberoEnv(
            task_suite=catalog,
            task_id=task_id,
            task_suite_name=suite,
            episode_length=episode_length,
            camera_name="agentview_image,robot0_eye_in_hand_image",
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            observation_width=256,
            observation_height=256,
            init_states=True,
            episode_index=0,
            n_envs=1,
            control_freq=fps,
            control_mode=control_mode,
            is_libero_plus=True,
            hard_reset=hard_reset,
        )

    return gym.vector.SyncVectorEnv(
        [factory], autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP
    )


def _group_success(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[field]), []).append(row)
    return {
        name: {
            "episodes": len(values),
            "successes": sum(bool(value["success"]) for value in values),
            "success_rate": float(np.mean([bool(value["success"]) for value in values])),
            "wilson_95": wilson_interval(
                sum(bool(value["success"]) for value in values), len(values)
            ),
        }
        for name, values in sorted(grouped.items())
    }


def summarize_libero_plus(rows: list[dict[str, Any]], *, full_benchmark: bool) -> dict[str, Any]:
    if not rows:
        raise ValueError("LIBERO-Plus evaluation contains no complete episodes")
    successes = sum(bool(row["success"]) for row in rows)
    latencies = [latency for row in rows for latency in row["model_latency_s"]]
    return {
        "full_benchmark": full_benchmark,
        "expected_full_task_count": TOTAL_TASKS,
        "episodes": len(rows),
        "successes": successes,
        "micro_success_rate": successes / len(rows),
        "micro_wilson_95": wilson_interval(successes, len(rows)),
        "per_suite": _group_success(rows, "suite"),
        "per_category": _group_success(rows, "category"),
        "per_difficulty": _group_success(rows, "difficulty_level"),
        "mean_model_latency_s": float(np.mean([row["mean_model_latency_s"] for row in rows])),
        "p95_replan_latency_s": float(np.quantile(latencies, 0.95)),
        "mean_environment_latency_s": float(
            np.mean([row["mean_environment_latency_s"] for row in rows])
        ),
        "mean_action_l2": float(np.mean([row["mean_action_l2"] for row in rows])),
        "mean_action_delta_l2": float(
            np.mean([row["mean_action_delta_l2"] for row in rows])
        ),
    }


def evaluate_libero_plus(
    config: dict[str, Any], output_dir: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    """Evaluate a policy with per-episode persistence and exact resume."""

    output = Path(output_dir)
    _prepare_output(output, config, resume=resume)
    evaluation = config["evaluation"]
    suites = [str(value) for value in evaluation["suites"]]
    requested_task_ids = evaluation.get("task_ids")
    sample_per_category = evaluation.get("sample_per_category")
    sample_seed = int(evaluation.get("sample_seed", 0))
    if requested_task_ids is not None and sample_per_category is not None:
        raise ValueError("LIBERO-Plus task_ids and sample_per_category are mutually exclusive")
    reset_seeds = [int(value) for value in evaluation["reset_seeds"]]
    full_benchmark = (
        len(suites) == len(SUITE_COUNTS)
        and set(suites) == set(SUITE_COUNTS)
        and requested_task_ids is None
        and sample_per_category is None
        and len(reset_seeds) == 1
    )

    initialization = tqdm(total=2, desc="initialize LIBERO-Plus", unit="stage", dynamic_ncols=True)
    initialization.set_postfix_str("verifying 10,030-task fork", refresh=True)
    catalogs, classification, classification_path = _load_catalogs(suites)
    initialization.update(1)
    initialization.set_postfix_str("loading policy", refresh=True)
    policy, identity = load_inference_policy(config)
    initialization.update(1)
    initialization.close()

    from lerobot.envs.configs import LiberoPlusEnv

    env_processor, _ = LiberoPlusEnv().get_env_processors()
    if sample_per_category is not None:
        selected = select_category_sample(
            classification,
            suites,
            per_category=int(sample_per_category),
            seed=sample_seed,
        )
    else:
        selected = {}
        for suite in suites:
            ids = (
                list(range(SUITE_COUNTS[suite]))
                if requested_task_ids is None
                else [int(value) for value in requested_task_ids]
            )
            if any(value >= SUITE_COUNTS[suite] for value in ids):
                raise ValueError(
                    f"LIBERO-Plus task id exceeds {suite} range [0,{SUITE_COUNTS[suite] - 1}]"
                )
            selected[suite] = ids
    requested = {
        (suite, task_id, seed)
        for suite, ids in selected.items()
        for task_id in ids
        for seed in reset_seeds
    }

    episodes_path = output / "episodes.jsonl"
    rows = _load_existing(episodes_path)
    completed = {
        (str(row["suite"]), int(row["task_id"]), int(row["reset_seed"])) for row in rows
    }
    if not completed.issubset(requested):
        raise ValueError("resume output contains episodes outside the current evaluation scope")
    manifest = {
        "format": "tmdpolicy.libero-plus-evaluation/v1",
        "policy": identity,
        "libero_plus_source": _source_identity(classification_path),
        "suite_counts": SUITE_COUNTS,
        "expected_total_tasks": TOTAL_TASKS,
        "full_benchmark": full_benchmark,
        "selection": {
            "mode": "category_sample" if sample_per_category is not None else (
                "task_ids" if requested_task_ids is not None else "full"
            ),
            "sample_per_category": sample_per_category,
            "sample_seed": sample_seed if sample_per_category is not None else None,
            "tasks": selected,
        },
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            previous_manifest.get("libero_plus_source") != manifest["libero_plus_source"]
            or previous_manifest.get("policy", {}).get("checkpoint_sha256")
            != identity["checkpoint_sha256"]
        ):
            raise ValueError("LIBERO-Plus resume source or policy checkpoint identity changed")
    else:
        _atomic_json(manifest, manifest_path)

    progress = tqdm(
        total=len(requested),
        initial=len(completed),
        desc="evaluate LIBERO-Plus",
        unit="episode",
        dynamic_ncols=True,
    )
    with episodes_path.open("a", encoding="utf-8") as handle:
        for suite in suites:
            catalog = catalogs[suite]
            for task_id in selected[suite]:
                metadata = classification[suite][task_id]
                task = catalog.tasks[task_id]
                for reset_seed in reset_seeds:
                    key = (suite, task_id, reset_seed)
                    if key in completed:
                        continue
                    progress.set_postfix_str(
                        f"{suite}:{task_id}/{SUITE_COUNTS[suite] - 1} seed={reset_seed}",
                        refresh=True,
                    )
                    env = _single_task_env(
                        catalog,
                        suite=suite,
                        task_id=task_id,
                        fps=int(evaluation["fps"]),
                        episode_length=int(evaluation["suite_max_episode_steps"][suite]),
                        control_mode=str(evaluation.get("control_mode", "relative")),
                        hard_reset=bool(evaluation.get("hard_reset", True)),
                    )
                    try:
                        metrics, _ = run_episode(
                            env,
                            env_processor,
                            policy,
                            instruction=str(task.language),
                            reset_seed=reset_seed,
                            task_index=SUITE_OFFSETS[suite] + task_id,
                            execution_horizon=int(config["horizons"]["execution"]),
                            max_steps=int(evaluation["suite_max_episode_steps"][suite]),
                            synchronize_cuda=bool(evaluation.get("synchronize_cuda", True)),
                            replan_metadata=None,
                            progress_description=f"LIBERO+ {suite}:{task_id}",
                        )
                    finally:
                        env.close()
                    row = {
                        "suite": suite,
                        "task_id": task_id,
                        "variant_id": int(metadata["id"]),
                        "global_task_index": SUITE_OFFSETS[suite] + task_id,
                        "task_name": str(task.name),
                        "instruction": str(task.language),
                        "category": str(metadata["category"]),
                        "difficulty_level": int(metadata["difficulty_level"]),
                        "reset_seed": reset_seed,
                        **metrics,
                    }
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                    handle.flush()
                    if (len(rows) + 1) % 10 == 0:
                        os.fsync(handle.fileno())
                    rows.append(row)
                    completed.add(key)
                    progress.update(1)
                    _atomic_json(
                        {
                            "completed_episodes": len(completed),
                            "requested_episodes": len(requested),
                            "fraction": len(completed) / len(requested),
                            "last_completed": list(key),
                        },
                        output / "progress.json",
                    )
    progress.close()
    summary = summarize_libero_plus(rows, full_benchmark=full_benchmark)
    report = {"policy": identity, "summary": summary}
    _atomic_json(report, output / "evaluation.json")
    return report


__all__ = [
    "SUITE_COUNTS",
    "TOTAL_TASKS",
    "CATEGORY_NAMES",
    "evaluate_libero_plus",
    "select_category_sample",
    "summarize_libero_plus",
]
