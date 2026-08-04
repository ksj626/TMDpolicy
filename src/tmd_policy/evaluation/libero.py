"""Complete-episode, paired-seed LIBERO evaluation and rollout collection."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tmd_policy.config import project_path, save_resolved_config
from tmd_policy.backends.lerobot.compatibility import verify_installed_lerobot
from tmd_policy.data.libero import load_episode_manifest
from tmd_policy.rollout import RolloutEpisode, RolloutStore

from .policy import InferencePolicy, load_inference_policy


_SUITE_OFFSETS = {"libero_spatial": 0, "libero_object": 10, "libero_goal": 20, "libero_10": 30}


def _canonical_task_identity(
    manifest: dict[str, Any], suite: str, local_task_id: int, instruction: str
) -> tuple[int, str]:
    """Resolve a suite-local environment task to the dataset's global task ID."""

    normalized = " ".join(instruction.split())
    matches = [
        int(index)
        for index, value in manifest["tasks"].items()
        if " ".join(str(value["instruction"]).split()) == normalized
    ]
    expected = _SUITE_OFFSETS.get(suite, -10_000) + local_task_id
    if expected in matches:
        canonical = expected
    elif len(matches) == 1:
        canonical = matches[0]
    else:
        raise RuntimeError(
            f"cannot uniquely map LIBERO environment task {suite}:{local_task_id} ({instruction!r}) "
            f"to the immutable dataset manifest; matches={matches}"
        )
    return canonical, str(manifest["tasks"][str(canonical)]["canonical_task_uid"])


def _canonical_observation(raw: dict[str, Any], env_processor: Any) -> dict[str, Any]:
    from lerobot.envs.utils import preprocess_observation

    return env_processor(preprocess_observation(raw))


def _visual(observation: dict[str, Any]) -> torch.Tensor:
    values = []
    for key in sorted(key for key in observation if key.startswith("observation.images.")):
        image = torch.as_tensor(observation[key]).float()
        if image.ndim == 3:
            image = image.unsqueeze(0)
        values.append(image.mean(dim=(-2, -1))[0].cpu())
        if len(values) == 2:
            break
    if not values:
        raise RuntimeError("LIBERO observation contains no images")
    while len(values) < 2:
        values.append(torch.zeros_like(values[0]))
    return torch.cat(values)


def _success(info: Any) -> bool:
    if isinstance(info, dict):
        if "is_success" in info:
            return bool(np.asarray(info["is_success"]).reshape(-1)[0])
        final = info.get("final_info")
        if final is not None:
            value = np.asarray(final, dtype=object).reshape(-1)[0]
            return bool(value.get("is_success", False)) if isinstance(value, dict) else False
    return False


def _sync(device: torch.device, enabled: bool) -> None:
    if enabled and device.type == "cuda":
        torch.cuda.synchronize(device)


def run_episode(
    env: Any,
    env_processor: Any,
    policy: InferencePolicy,
    *,
    instruction: str,
    reset_seed: int,
    task_index: int,
    execution_horizon: int,
    max_steps: int,
    synchronize_cuda: bool,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION

    policy.reset()
    raw, info = env.reset(seed=[reset_seed], options={NEW_ROLLOUT_OPTION: True})
    observation = _canonical_observation(raw, env_processor)
    states = [torch.as_tensor(observation["observation.state"])[0, :8].float().cpu()]
    actions: list[torch.Tensor] = []
    visuals: list[torch.Tensor] = []
    model_latencies: list[float] = []
    environment_latencies: list[float] = []
    successful = _success(info)
    terminated_value = truncated_value = False
    replans = 0
    while len(actions) < max_steps and not (terminated_value or truncated_value):
        model_batch = dict(observation)
        model_batch["task"] = [instruction]
        seed = 100_000_007 * task_index + 10_007 * reset_seed + replans
        _sync(policy.student.device, synchronize_cuda)
        started = time.perf_counter()
        plan = policy.plan(model_batch, noise_seed=seed)[0].detach().cpu()
        _sync(policy.student.device, synchronize_cuda)
        model_latencies.append(time.perf_counter() - started)
        for action in plan[:execution_horizon]:
            if len(actions) >= max_steps:
                truncated_value = True
                break
            visuals.append(_visual(observation))
            environment_started = time.perf_counter()
            raw, _, terminated, truncated, info = env.step(action.numpy()[None])
            environment_latencies.append(time.perf_counter() - environment_started)
            actions.append(action.float())
            observation = _canonical_observation(raw, env_processor)
            states.append(torch.as_tensor(observation["observation.state"])[0, :8].float().cpu())
            terminated_value = bool(np.asarray(terminated).reshape(-1)[0])
            truncated_value = bool(np.asarray(truncated).reshape(-1)[0])
            successful = successful or _success(info)
            if terminated_value or truncated_value:
                break
        replans += 1
    if len(actions) >= max_steps and not terminated_value:
        truncated_value = True
    action_path = torch.stack(actions)
    delta_norm = (
        torch.linalg.vector_norm(action_path[1:] - action_path[:-1], dim=-1)
        if len(actions) > 1
        else torch.zeros(1)
    )
    payload = {
        "states": torch.stack(states),
        "actions": action_path,
        "visual": torch.stack(visuals),
    }
    metrics = {
        "success": successful,
        "terminated": terminated_value,
        "truncated": truncated_value,
        "steps": len(actions),
        "replans": replans,
        "model_latency_s": model_latencies,
        "environment_latency_s": environment_latencies,
        "mean_model_latency_s": float(np.mean(model_latencies)),
        "p95_model_latency_s": float(np.quantile(model_latencies, 0.95)),
        "mean_environment_latency_s": float(np.mean(environment_latencies)),
        "mean_action_l2": float(torch.linalg.vector_norm(action_path, dim=-1).mean()),
        "mean_action_delta_l2": float(delta_norm.mean()),
    }
    return metrics, payload


def wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> list[float]:
    if count < 1:
        return [float("nan"), float("nan")]
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    margin = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("evaluation contains no complete episodes")
    groups: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        groups.setdefault(f"{episode['suite']}:{episode['task_id']}", []).append(episode)
    per_task = {}
    for key, values in sorted(groups.items()):
        successes = sum(bool(value["success"]) for value in values)
        per_task[key] = {
            "successes": successes,
            "episodes": len(values),
            "success_rate": successes / len(values),
            "wilson_95": wilson_interval(successes, len(values)),
        }
    total_success = sum(value["successes"] for value in per_task.values())
    return {
        "episodes": len(episodes),
        "micro_success_rate": total_success / len(episodes),
        "micro_wilson_95": wilson_interval(total_success, len(episodes)),
        "macro_task_success_rate": float(np.mean([value["success_rate"] for value in per_task.values()])),
        "per_task": per_task,
        "mean_model_latency_s": float(np.mean([value["mean_model_latency_s"] for value in episodes])),
        "p95_replan_latency_s": float(
            np.quantile([latency for value in episodes for latency in value["model_latency_s"]], 0.95)
        ),
        "mean_environment_latency_s": float(
            np.mean([value["mean_environment_latency_s"] for value in episodes])
        ),
        "mean_action_l2": float(np.mean([value["mean_action_l2"] for value in episodes])),
        "mean_action_delta_l2": float(
            np.mean([value["mean_action_delta_l2"] for value in episodes])
        ),
    }


def evaluate_libero(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_resolved_config(config, output / "resolved_config.yaml")
    policy, identity = load_inference_policy(config)
    evaluation = config["evaluation"]
    manifest = load_episode_manifest(project_path(config["dataset"]["manifest"]))
    from lerobot.envs.configs import LiberoEnv

    episodes: list[dict[str, Any]] = []
    rollout_store = None
    if bool(evaluation.get("save_rollouts", False)):
        rollout_store = RolloutStore(output / "rollouts")
        rollout_store.initialize(
            {
                "purpose": "evaluation",
                "policy": identity,
                "dataset_revision": config["dataset"]["revision"],
            }
        )
    for suite_spec in evaluation["benchmark"]:
        suite = suite_spec["suite"]
        task_ids = [int(value) for value in suite_spec["task_ids"]]
        env_config = LiberoEnv(
            task=suite,
            task_ids=task_ids,
            fps=int(evaluation.get("fps", 10)),
            observation_height=256,
            observation_width=256,
            episode_length=int(evaluation["max_episode_steps"]),
        )
        env_processor, _ = env_config.get_env_processors()
        environments = env_config.create_envs(n_envs=1, use_async_envs=False)
        try:
            for task_id in task_ids:
                env = environments[suite][task_id]
                instruction = str(env.call("task_description")[0])
                dataset_task_index, task_uid = _canonical_task_identity(
                    manifest, suite, task_id, instruction
                )
                for reset_seed in evaluation["reset_seeds"]:
                    metrics, payload = run_episode(
                        env,
                        env_processor,
                        policy,
                        instruction=instruction,
                        reset_seed=int(reset_seed),
                        task_index=dataset_task_index,
                        execution_horizon=int(config["horizons"]["execution"]),
                        max_steps=int(evaluation["max_episode_steps"]),
                        synchronize_cuda=bool(evaluation.get("synchronize_cuda", True)),
                    )
                    row = {
                        "suite": suite,
                        "task_id": task_id,
                        "dataset_task_index": dataset_task_index,
                        "canonical_task_uid": task_uid,
                        "instruction": instruction,
                        "reset_seed": int(reset_seed),
                        **metrics,
                    }
                    episodes.append(row)
                    if rollout_store is not None:
                        rollout_store.append(
                            RolloutEpisode(
                                **payload,
                                task_index=dataset_task_index,
                                canonical_task_uid=row["canonical_task_uid"],
                                instruction=instruction,
                                reset_seed=int(reset_seed),
                                success=bool(metrics["success"]),
                                terminated=bool(metrics["terminated"]),
                                truncated=bool(metrics["truncated"]),
                                policy_checkpoint=identity["checkpoint"],
                                policy_checkpoint_sha256=identity["checkpoint_sha256"],
                                collection_round=0,
                                split="test",
                            )
                        )
        finally:
            for group in environments.values():
                for env in group.values():
                    env.close()
    report = {
        "data": "real complete-episode LIBERO",
        "policy": identity,
        "paired_keys": ["suite", "task_id", "reset_seed"],
        "benchmark": evaluation["benchmark"],
        "reset_seeds": evaluation["reset_seeds"],
        "episodes": episodes,
        "summary": summarize(episodes),
        "provenance": {
            "lerobot": verify_installed_lerobot(
                expected_source_hashes=config["backend"].get("expected_source_hashes")
            ),
            "models": config["models"],
            "dataset": config["dataset"],
        },
    }
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def collect_student_rollouts(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    policy, identity = load_inference_policy(config)
    collection = config["collection"]
    manifest = load_episode_manifest(project_path(config["dataset"]["manifest"]))
    from lerobot.envs.configs import LiberoEnv

    store = RolloutStore(output)
    store.initialize(
        {
            "purpose": "occupancy_student",
            "policy": identity,
            "dataset_revision": config["dataset"]["revision"],
            "collection_round": collection["collection_round"],
            "mode": "current_policy_at_collection_time",
            "lerobot": verify_installed_lerobot(
                expected_source_hashes=config["backend"].get("expected_source_hashes")
            ),
            "models": config["models"],
        }
    )
    save_resolved_config(config, output / "resolved_config.yaml")
    env_config = LiberoEnv(
        task=collection["suite"],
        task_ids=collection["task_ids"],
        fps=int(collection.get("fps", 10)),
        observation_height=256,
        observation_width=256,
        episode_length=int(collection["max_episode_steps"]),
    )
    env_processor, _ = env_config.get_env_processors()
    environments = env_config.create_envs(n_envs=1, use_async_envs=False)
    try:
        for task_id in collection["task_ids"]:
            env = environments[collection["suite"]][task_id]
            instruction = str(env.call("task_description")[0])
            dataset_task_index, task_uid = _canonical_task_identity(
                manifest, collection["suite"], int(task_id), instruction
            )
            for split, seeds in (
                ("train", collection["train_reset_seeds"]),
                ("validation", collection["validation_reset_seeds"]),
            ):
                for reset_seed in seeds:
                    metrics, payload = run_episode(
                        env,
                        env_processor,
                        policy,
                        instruction=instruction,
                        reset_seed=int(reset_seed),
                        task_index=dataset_task_index,
                        execution_horizon=int(config["horizons"]["execution"]),
                        max_steps=int(collection["max_episode_steps"]),
                        synchronize_cuda=True,
                    )
                    store.append(
                        RolloutEpisode(
                            **payload,
                            task_index=dataset_task_index,
                            canonical_task_uid=task_uid,
                            instruction=instruction,
                            reset_seed=int(reset_seed),
                            success=bool(metrics["success"]),
                            terminated=bool(metrics["terminated"]),
                            truncated=bool(metrics["truncated"]),
                            policy_checkpoint=identity["checkpoint"],
                            policy_checkpoint_sha256=identity["checkpoint_sha256"],
                            collection_round=int(collection["collection_round"]),
                            split=split,
                        )
                    )
    finally:
        for group in environments.values():
            for env in group.values():
                env.close()
    report = store.validate()
    (output / "collection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = ["collect_student_rollouts", "evaluate_libero", "run_episode", "summarize", "wilson_interval"]
