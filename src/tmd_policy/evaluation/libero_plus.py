"""Resumable serial or multi-GPU-sharded LIBERO-Plus evaluation."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import inspect
import io
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import yaml
from tqdm.auto import tqdm

from tmd_policy.config import save_resolved_config

from .libero import run_episode_batch, wilson_interval
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


def resolve_video_setting(
    *, sample_per_category: int | None, save_videos: bool | None
) -> bool:
    """Default videos on for category samples while keeping full runs opt-in."""

    return bool(sample_per_category is not None) if save_videos is None else save_videos


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
    evaluation.pop("batch_size", None)
    evaluation.pop("devices", None)
    evaluation.pop("parallel_worker", None)
    evaluation.pop("task_map", None)
    evaluation.setdefault("save_videos", None)
    evaluation.setdefault("video_camera", "image")
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


def _select_tasks(
    evaluation: dict[str, Any],
    classification: dict[str, list[dict[str, Any]]],
    suites: list[str],
) -> dict[str, list[int]]:
    task_map = evaluation.get("task_map")
    if task_map is not None:
        return {
            suite: [int(value) for value in task_map.get(suite, [])]
            for suite in suites
        }
    requested_task_ids = evaluation.get("task_ids")
    sample_per_category = evaluation.get("sample_per_category")
    if sample_per_category is not None:
        return select_category_sample(
            classification,
            suites,
            per_category=int(sample_per_category),
            seed=int(evaluation.get("sample_seed", 0)),
        )
    selected: dict[str, list[int]] = {}
    for suite in suites:
        ids = (
            list(range(SUITE_COUNTS[suite]))
            if requested_task_ids is None
            else [int(value) for value in requested_task_ids]
        )
        if any(value < 0 or value >= SUITE_COUNTS[suite] for value in ids):
            raise ValueError(
                f"LIBERO-Plus task id exceeds {suite} range [0,{SUITE_COUNTS[suite] - 1}]"
            )
        selected[suite] = ids
    return selected


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


def _evaluate_libero_plus_serial(
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
    batch_size = int(evaluation.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("LIBERO-Plus batch_size must be positive")
    save_videos = resolve_video_setting(
        sample_per_category=(
            int(sample_per_category) if sample_per_category is not None else None
        ),
        save_videos=evaluation.get("save_videos"),
    )
    if requested_task_ids is not None and sample_per_category is not None:
        raise ValueError("LIBERO-Plus task_ids and sample_per_category are mutually exclusive")
    reset_seeds = [int(value) for value in evaluation["reset_seeds"]]
    full_benchmark = (
        len(suites) == len(SUITE_COUNTS)
        and set(suites) == set(SUITE_COUNTS)
        and requested_task_ids is None
        and sample_per_category is None
        and evaluation.get("task_map") is None
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
    selected = _select_tasks(evaluation, classification, suites)
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
            "mode": "worker_shard" if evaluation.get("task_map") is not None else (
                "category_sample" if sample_per_category is not None else (
                "task_ids" if requested_task_ids is not None else (
                    "full_benchmark" if full_benchmark else "full_suite"
                )
            )),
            "sample_per_category": sample_per_category,
            "sample_seed": sample_seed if sample_per_category is not None else None,
            "tasks": selected,
        },
        "batch_size": batch_size,
        "save_videos": save_videos,
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
            pending = [
                (task_id, reset_seed)
                for task_id in selected[suite]
                for reset_seed in reset_seeds
                if (suite, task_id, reset_seed) not in completed
            ]
            for chunk_start in range(0, len(pending), batch_size):
                chunk = pending[chunk_start : chunk_start + batch_size]
                progress.set_postfix_str(
                    f"{suite} batch={len(chunk)} tasks={chunk[0][0]}..{chunk[-1][0]}",
                    refresh=True,
                )
                envs: list[Any] = []
                specs: list[dict[str, Any]] = []
                try:
                    for task_id, reset_seed in chunk:
                        task = catalog.tasks[task_id]
                        metadata = classification[suite][task_id]
                        env = _single_task_env(
                            catalog,
                            suite=suite,
                            task_id=task_id,
                            fps=int(evaluation["fps"]),
                            episode_length=int(evaluation["suite_max_episode_steps"][suite]),
                            control_mode=str(evaluation.get("control_mode", "relative")),
                            hard_reset=bool(evaluation.get("hard_reset", True)),
                        )
                        envs.append(env)
                        video_path = (
                            output
                            / "videos"
                            / suite
                            / f"task-{task_id:04d}_variant-{int(metadata['id']):04d}_seed-{reset_seed}.mp4"
                            if save_videos
                            else None
                        )
                        specs.append(
                            {
                                "env": env,
                                "instruction": str(task.language),
                                "reset_seed": reset_seed,
                                "task_index": SUITE_OFFSETS[suite] + task_id,
                                "replan_metadata": None,
                                "video_path": video_path,
                            }
                        )
                    results = run_episode_batch(
                        specs,
                        env_processor,
                        policy,
                        execution_horizon=int(config["horizons"]["execution"]),
                        max_steps=int(evaluation["suite_max_episode_steps"][suite]),
                        synchronize_cuda=bool(evaluation.get("synchronize_cuda", True)),
                        progress_description=f"LIBERO+ {suite} batch {chunk_start // batch_size + 1}",
                        video_fps=int(evaluation["fps"]),
                        video_camera=str(evaluation.get("video_camera", "image")),
                    )
                finally:
                    for env in envs:
                        env.close()
                for (task_id, reset_seed), (metrics, payload) in zip(
                    chunk, results, strict=True
                ):
                    metadata = classification[suite][task_id]
                    task = catalog.tasks[task_id]
                    key = (suite, task_id, reset_seed)
                    video_path = payload.get("video_path")
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
                        "video": (
                            str(Path(video_path).relative_to(output)) if video_path else None
                        ),
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


def _evaluate_libero_plus_multi_gpu(
    config: dict[str, Any], output_dir: str | Path, *, resume: bool
) -> dict[str, Any]:
    """Shard tasks across policy replicas; each worker keeps the serial loop."""

    output = Path(output_dir)
    _prepare_output(output, config, resume=resume)
    evaluation = config["evaluation"]
    devices = [str(value) for value in evaluation["devices"]]
    suites = [str(value) for value in evaluation["suites"]]
    _, classification, classification_path = _load_catalogs(suites)
    selected = _select_tasks(evaluation, classification, suites)
    task_pairs = [
        (suite, task_id)
        for suite in suites
        for task_id in selected[suite]
    ]
    if not task_pairs:
        raise ValueError("LIBERO-Plus multi-GPU selection contains no tasks")
    worker_count = min(len(devices), len(task_pairs))
    shards: list[list[tuple[str, int]]] = [[] for _ in range(worker_count)]
    for index, pair in enumerate(task_pairs):
        shards[index % worker_count].append(pair)

    workers_root = output / "parallel_workers"
    workers_root.mkdir(exist_ok=True)
    processes: list[tuple[int, str, Path, subprocess.Popen[str], Any]] = []
    for worker_index, (device, shard) in enumerate(
        zip(devices[:worker_count], shards, strict=True)
    ):
        worker_name = f"worker-{worker_index:02d}"
        worker_output = workers_root / worker_name
        worker_config = copy.deepcopy(config)
        worker_config["classification"] = (
            f"{config['classification']}; serial multi-GPU shard {worker_index}/{worker_count}"
        )
        worker_config["policy"]["device"] = device
        worker_evaluation = worker_config["evaluation"]
        worker_evaluation["devices"] = []
        worker_evaluation["parallel_worker"] = True
        worker_evaluation["batch_size"] = 1
        worker_evaluation["task_ids"] = None
        worker_evaluation["sample_per_category"] = None
        worker_evaluation["save_videos"] = resolve_video_setting(
            sample_per_category=(
                int(evaluation["sample_per_category"])
                if evaluation.get("sample_per_category") is not None
                else None
            ),
            save_videos=evaluation.get("save_videos"),
        )
        task_map = {
            suite: [task_id for selected_suite, task_id in shard if selected_suite == suite]
            for suite in suites
        }
        worker_evaluation["suites"] = [suite for suite in suites if task_map[suite]]
        worker_evaluation["task_map"] = task_map
        worker_config["output"]["directory"] = str(worker_output)
        config_path = workers_root / f"{worker_name}.yaml"
        save_resolved_config(worker_config, config_path)
        log_path = workers_root / f"{worker_name}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "tmd_policy.cli",
            "evaluate",
            "libero-plus",
            "--config",
            str(config_path),
            "--output",
            str(worker_output),
        ]
        if worker_output.exists():
            command.append("--resume")
        environment = dict(os.environ)
        environment.setdefault("MUJOCO_GL", "egl")
        # EGL and CUDA device enumerations are not necessarily the same (this
        # container exposes one EGL render device and many CUDA devices). Keep
        # rendering on the launcher-selected/default EGL device; only the VLA
        # replica is assigned to `device` above.
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((worker_index, device, worker_output, process, log_handle))

    reset_seeds = [int(value) for value in evaluation["reset_seeds"]]
    total_episodes = len(task_pairs) * len(reset_seeds)
    progress = tqdm(
        total=total_episodes,
        desc=f"LIBERO-Plus {worker_count}-GPU shards",
        unit="episode",
        dynamic_ncols=True,
    )
    try:
        while any(process.poll() is None for _, _, _, process, _ in processes):
            completed = 0
            for _, _, worker_output, _, _ in processes:
                progress_path = worker_output / "progress.json"
                if progress_path.exists():
                    try:
                        completed += int(
                            json.loads(progress_path.read_text(encoding="utf-8"))[
                                "completed_episodes"
                            ]
                        )
                    except (OSError, ValueError, KeyError, json.JSONDecodeError):
                        pass
            progress.n = min(completed, total_episodes)
            progress.set_postfix(alive=sum(p.poll() is None for *_, p, _ in processes))
            progress.refresh()
            time.sleep(1.0)
    finally:
        progress.close()
        for *_, handle in processes:
            handle.close()

    failures = [
        (index, device, process.returncode)
        for index, device, _, process, _ in processes
        if process.returncode != 0
    ]
    if failures:
        details = []
        for index, device, return_code in failures:
            log_path = workers_root / f"worker-{index:02d}.log"
            tail = "\n".join(log_path.read_text(encoding="utf-8").splitlines()[-30:])
            details.append(f"worker {index} ({device}) exited {return_code}:\n{tail}")
        raise RuntimeError("LIBERO-Plus parallel evaluation failed\n" + "\n".join(details))

    rows: list[dict[str, Any]] = []
    worker_manifests: list[dict[str, Any]] = []
    for worker_index, _, worker_output, _, _ in processes:
        worker_rows = _load_existing(worker_output / "episodes.jsonl")
        for row in worker_rows:
            if row.get("video"):
                row["video"] = str(
                    Path("parallel_workers")
                    / f"worker-{worker_index:02d}"
                    / str(row["video"])
                )
            rows.append(row)
        worker_manifests.append(
            json.loads((worker_output / "manifest.json").read_text(encoding="utf-8"))
        )
    suite_order = {suite: index for index, suite in enumerate(suites)}
    rows.sort(key=lambda row: (suite_order[str(row["suite"])], int(row["task_id"]), int(row["reset_seed"])))
    expected = {
        (suite, task_id, seed)
        for suite, ids in selected.items()
        for task_id in ids
        for seed in reset_seeds
    }
    actual = {
        (str(row["suite"]), int(row["task_id"]), int(row["reset_seed"]))
        for row in rows
    }
    if actual != expected:
        raise RuntimeError(
            f"parallel LIBERO-Plus shards are incomplete: missing={len(expected - actual)}, "
            f"unexpected={len(actual - expected)}"
        )
    episodes_path = output / "episodes.jsonl"
    episodes_partial = episodes_path.with_suffix(".jsonl.partial")
    episodes_partial.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(episodes_partial, episodes_path)
    full_benchmark = (
        set(suites) == set(SUITE_COUNTS)
        and evaluation.get("task_ids") is None
        and evaluation.get("sample_per_category") is None
        and len(reset_seeds) == 1
    )
    manifest = {
        "format": "tmdpolicy.libero-plus-evaluation/v1",
        "policy": worker_manifests[0]["policy"],
        "libero_plus_source": _source_identity(classification_path),
        "suite_counts": SUITE_COUNTS,
        "expected_total_tasks": TOTAL_TASKS,
        "full_benchmark": full_benchmark,
        "selection": {
            "mode": (
                "category_sample"
                if evaluation.get("sample_per_category") is not None
                else "task_ids"
                if evaluation.get("task_ids") is not None
                else "full_benchmark"
                if full_benchmark
                else "full_suite"
            ),
            "sample_per_category": evaluation.get("sample_per_category"),
            "sample_seed": (
                int(evaluation.get("sample_seed", 0))
                if evaluation.get("sample_per_category") is not None
                else None
            ),
            "tasks": selected,
        },
        "parallelism": {
            "mode": "independent_serial_processes",
            "devices": devices[:worker_count],
            "workers": worker_count,
            "per_worker_batch_size": 1,
        },
        "save_videos": resolve_video_setting(
            sample_per_category=(
                int(evaluation["sample_per_category"])
                if evaluation.get("sample_per_category") is not None
                else None
            ),
            save_videos=evaluation.get("save_videos"),
        ),
    }
    _atomic_json(manifest, output / "manifest.json")
    _atomic_json(
        {
            "completed_episodes": len(rows),
            "requested_episodes": len(expected),
            "fraction": 1.0,
        },
        output / "progress.json",
    )
    summary = summarize_libero_plus(rows, full_benchmark=full_benchmark)
    report = {"policy": manifest["policy"], "summary": summary}
    _atomic_json(report, output / "evaluation.json")
    return report


def evaluate_libero_plus(
    config: dict[str, Any], output_dir: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    devices = [str(value) for value in config["evaluation"].get("devices", [])]
    if len(devices) > 1 and not bool(config["evaluation"].get("parallel_worker", False)):
        return _evaluate_libero_plus_multi_gpu(config, output_dir, resume=resume)
    if len(devices) == 1:
        config["policy"]["device"] = devices[0]
    return _evaluate_libero_plus_serial(config, output_dir, resume=resume)


__all__ = [
    "SUITE_COUNTS",
    "TOTAL_TASKS",
    "CATEGORY_NAMES",
    "evaluate_libero_plus",
    "resolve_video_setting",
    "select_category_sample",
    "summarize_libero_plus",
]
