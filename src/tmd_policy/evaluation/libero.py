"""Complete-episode, paired-seed LIBERO evaluation and rollout collection."""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from tmd_policy.config import project_path, save_resolved_config
from tmd_policy.backends.lerobot.compatibility import verify_installed_lerobot
from tmd_policy.data.libero import load_episode_manifest
from tmd_policy.libero_protocol import init_state_index_for_trial, set_fixed_init_state_index
from tmd_policy.rollout import ReplanRecord, RolloutEpisode, RolloutStore

from .policy import InferencePolicy, PI05InferencePolicy, load_inference_policy


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


def _stored_observations(
    observation: dict[str, Any]
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    images = {
        key: torch.as_tensor(value).detach().cpu().clone()
        for key, value in observation.items()
        if key.startswith("observation.images.") and not key.endswith("_is_pad")
    }
    if not images:
        raise RuntimeError("LIBERO observation contains no canonical camera images")
    metadata = {}
    for key, value in images.items():
        layouts = {3: "CHW", 4: "BCHW"}
        metadata[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "layout": layouts.get(value.ndim, f"rank-{value.ndim}"),
            "encoding": "lossless torch tensor",
        }
    return images, metadata


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
    policy: InferencePolicy | PI05InferencePolicy,
    *,
    instruction: str,
    reset_seed: int,
    init_state_index: int | None = None,
    task_index: int,
    execution_horizon: int,
    max_steps: int,
    synchronize_cuda: bool,
    replan_metadata: dict[str, Any] | None,
    progress_description: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION

    policy.reset()
    resolved_init_state = set_fixed_init_state_index(
        env,
        reset_seed if init_state_index is None else init_state_index,
    )
    raw, info = env.reset(seed=[reset_seed], options={NEW_ROLLOUT_OPTION: True})
    observation = _canonical_observation(raw, env_processor)
    actions: list[torch.Tensor] = []
    replan_records: list[ReplanRecord] = []
    model_latencies: list[float] = []
    environment_latencies: list[float] = []
    successful = _success(info)
    terminated_value = truncated_value = False
    replans = 0
    step_progress = tqdm(
        total=max_steps,
        desc=progress_description or "LIBERO episode",
        unit="env-step",
        leave=False,
        dynamic_ncols=True,
        disable=progress_description is None,
    )
    while len(actions) < max_steps and not (terminated_value or truncated_value):
        model_batch = dict(observation)
        model_batch["task"] = [instruction]
        seed = 100_000_007 * task_index + 10_007 * reset_seed + replans
        start_step = len(actions)
        if replan_metadata is not None:
            start_state = (
                torch.as_tensor(observation["observation.state"])[0, :8]
                .float()
                .cpu()
                .clone()
            )
            stored_images, image_metadata = _stored_observations(observation)
        _sync(policy.device, synchronize_cuda)
        started = time.perf_counter()
        plan_progress = tqdm(
            total=policy.inference_work_units,
            desc=f"{progress_description or 'LIBERO episode'} plan {replans + 1}",
            unit="NFE",
            leave=False,
            dynamic_ncols=True,
            disable=progress_description is None,
        )
        try:
            plan = policy.plan(
                model_batch,
                noise_seed=seed,
                step_callback=lambda: plan_progress.update(1),
            )[0].detach().cpu()
        finally:
            plan_progress.close()
        _sync(policy.device, synchronize_cuda)
        model_latencies.append(time.perf_counter() - started)
        if plan.shape != (50, 7):
            raise RuntimeError(f"policy plan must be canonical [50,7], got {tuple(plan.shape)}")
        executed_this_plan: list[torch.Tensor] = []
        for action in plan[:execution_horizon]:
            if len(actions) >= max_steps:
                truncated_value = True
                break
            environment_started = time.perf_counter()
            raw, _, terminated, truncated, info = env.step(action.numpy()[None])
            environment_latencies.append(time.perf_counter() - environment_started)
            actions.append(action.float())
            executed_this_plan.append(action.float())
            step_progress.update(1)
            observation = _canonical_observation(raw, env_processor)
            terminated_value = bool(np.asarray(terminated).reshape(-1)[0])
            truncated_value = bool(np.asarray(truncated).reshape(-1)[0])
            if len(actions) >= max_steps and not terminated_value:
                truncated_value = True
            successful = successful or _success(info)
            if terminated_value or truncated_value:
                break
        executed_tensor = (
            torch.stack(executed_this_plan)
            if executed_this_plan
            else torch.empty(0, 7, dtype=plan.dtype)
        )
        if replan_metadata is not None:
            replan_records.append(
                ReplanRecord(
                    suite=str(replan_metadata["suite"]),
                    suite_task_id=int(replan_metadata["suite_task_id"]),
                    global_task_index=task_index,
                    canonical_task_uid=str(replan_metadata["canonical_task_uid"]),
                    instruction=instruction,
                    reset_seed=reset_seed,
                    init_state_index=resolved_init_state,
                    policy_checkpoint=str(replan_metadata["policy_checkpoint"]),
                    policy_checkpoint_sha256=str(replan_metadata["policy_checkpoint_sha256"]),
                    policy_version=str(replan_metadata["policy_version"]),
                    collection_round=int(replan_metadata["collection_round"]),
                    environment_step=start_step,
                    state=start_state,
                    observations=stored_images,
                    observation_metadata=image_metadata,
                    planned_actions=plan.float(),
                    executed_prefix_length=len(executed_this_plan),
                    executed_actions=executed_tensor,
                    terminated=terminated_value,
                    truncated=truncated_value,
                    success=successful,
                    model_revision=str(replan_metadata["model_revision"]),
                    processor_revision=str(replan_metadata["processor_revision"]),
                    dataset_revision=str(replan_metadata["dataset_revision"]),
                )
            )
        replans += 1
        step_progress.set_postfix(replans=replans, success=successful, refresh=True)
    step_progress.close()
    if len(actions) >= max_steps and not terminated_value:
        truncated_value = True
    action_path = torch.stack(actions)
    delta_norm = (
        torch.linalg.vector_norm(action_path[1:] - action_path[:-1], dim=-1)
        if len(actions) > 1
        else torch.zeros(1)
    )
    payload = {
        "replans": tuple(replan_records),
        "init_state_index": resolved_init_state,
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


def _stack_observations(observations: list[dict[str, Any]]) -> dict[str, Any]:
    keys = set(observations[0])
    if any(set(observation) != keys for observation in observations):
        raise RuntimeError("batched LIBERO observations have different canonical keys")
    batch: dict[str, Any] = {}
    for key in sorted(keys):
        # Policy preprocessing consumes only canonical observation fields plus
        # the task strings attached below. Transition bookkeeping fields such
        # as info/action/reward are per-environment and must not be merged.
        if not key.startswith("observation."):
            continue
        values = [observation[key] for observation in observations]
        first = values[0]
        if first is None and all(value is None for value in values):
            batch[key] = None
        elif isinstance(first, torch.Tensor):
            batch[key] = torch.cat([torch.as_tensor(value) for value in values], dim=0)
        elif isinstance(first, np.ndarray):
            batch[key] = np.concatenate([np.asarray(value) for value in values], axis=0)
        elif isinstance(first, list):
            batch[key] = [item for value in values for item in value]
        else:
            raise TypeError(f"cannot batch canonical observation {key!r} of type {type(first)}")
    return batch


def _video_frame(raw: dict[str, Any], camera: str) -> np.ndarray:
    pixels = raw.get("pixels")
    if not isinstance(pixels, dict) or camera not in pixels:
        raise RuntimeError(
            f"LIBERO video camera {camera!r} is absent; available={sorted(pixels or {})}"
        )
    frame = np.asarray(pixels[camera])
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise RuntimeError(f"LIBERO video frame must be HWC RGB, got {frame.shape}")
    frame = np.ascontiguousarray(frame[::-1, ::-1])
    return frame.astype(np.uint8, copy=False)


def _open_video(path: Path | None, *, fps: int) -> Any | None:
    if path is None:
        return None
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=8)


def run_episode_batch(
    episode_specs: list[dict[str, Any]],
    env_processor: Any,
    policy: InferencePolicy | PI05InferencePolicy,
    *,
    execution_horizon: int,
    max_steps: int,
    synchronize_cuda: bool,
    progress_description: str | None = None,
    video_fps: int,
    video_camera: str = "image",
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Run independent episodes with only policy inference batched together."""

    from lerobot.envs.utils import NEW_ROLLOUT_OPTION

    if not episode_specs:
        return []
    policy.reset()
    states: list[dict[str, Any]] = []
    writers: list[Any | None] = []
    progress = tqdm(
        total=max_steps * len(episode_specs),
        desc=progress_description or "LIBERO episode batch",
        unit="env-step",
        leave=False,
        dynamic_ncols=True,
        disable=progress_description is None,
    )
    try:
        for spec in episode_specs:
            env = spec["env"]
            resolved_init_state = set_fixed_init_state_index(
                env,
                int(spec.get("init_state_index", spec["reset_seed"])),
            )
            raw, info = env.reset(
                seed=[int(spec["reset_seed"])], options={NEW_ROLLOUT_OPTION: True}
            )
            video_path = Path(spec["video_path"]) if spec.get("video_path") else None
            writer = _open_video(video_path, fps=video_fps)
            writers.append(writer)
            if writer is not None:
                writer.append_data(_video_frame(raw, video_camera))
            states.append(
                {
                    "spec": spec,
                    "observation": _canonical_observation(raw, env_processor),
                    "actions": [],
                    "replan_records": [],
                    "model_latencies": [],
                    "environment_latencies": [],
                    "successful": _success(info),
                    "terminated": False,
                    "truncated": False,
                    "replans": 0,
                    "video_path": str(video_path) if video_path is not None else None,
                    "video_writer": writer,
                    "init_state_index": resolved_init_state,
                }
            )

        while True:
            active = [
                state
                for state in states
                if len(state["actions"]) < max_steps
                and not (state["terminated"] or state["truncated"])
            ]
            if not active:
                break
            model_batch = _stack_observations([state["observation"] for state in active])
            model_batch["task"] = [str(state["spec"]["instruction"]) for state in active]
            seeds = [
                100_000_007 * int(state["spec"]["task_index"])
                + 10_007 * int(state["spec"]["reset_seed"])
                + int(state["replans"])
                for state in active
            ]
            starts: list[tuple[int, torch.Tensor | None, dict[str, torch.Tensor] | None, dict[str, dict[str, Any]] | None]] = []
            for state in active:
                metadata = state["spec"].get("replan_metadata")
                if metadata is None:
                    starts.append((len(state["actions"]), None, None, None))
                else:
                    observation = state["observation"]
                    stored_images, image_metadata = _stored_observations(observation)
                    starts.append(
                        (
                            len(state["actions"]),
                            torch.as_tensor(observation["observation.state"])[0, :8]
                            .float()
                            .cpu()
                            .clone(),
                            stored_images,
                            image_metadata,
                        )
                    )
            _sync(policy.device, synchronize_cuda)
            started = time.perf_counter()
            plan_progress = tqdm(
                total=policy.inference_work_units,
                desc=f"{progress_description or 'LIBERO batch'} plan",
                unit="NFE",
                leave=False,
                dynamic_ncols=True,
                disable=progress_description is None,
            )
            try:
                plans = policy.plan(
                    model_batch,
                    noise_seed=seeds,
                    step_callback=lambda: plan_progress.update(1),
                ).detach().cpu()
            finally:
                plan_progress.close()
            _sync(policy.device, synchronize_cuda)
            elapsed = time.perf_counter() - started
            if plans.shape != (len(active), 50, 7):
                raise RuntimeError(
                    f"batched policy plans must be [{len(active)},50,7], got {tuple(plans.shape)}"
                )
            for state, plan, start in zip(active, plans, starts, strict=True):
                state["model_latencies"].append(elapsed)
                executed: list[torch.Tensor] = []
                for action in plan[:execution_horizon]:
                    if len(state["actions"]) >= max_steps:
                        state["truncated"] = True
                        break
                    environment_started = time.perf_counter()
                    raw, _, terminated, truncated, info = state["spec"]["env"].step(
                        action.numpy()[None]
                    )
                    state["environment_latencies"].append(
                        time.perf_counter() - environment_started
                    )
                    state["actions"].append(action.float())
                    executed.append(action.float())
                    progress.update(1)
                    if state["video_writer"] is not None:
                        state["video_writer"].append_data(_video_frame(raw, video_camera))
                    state["observation"] = _canonical_observation(raw, env_processor)
                    state["terminated"] = bool(np.asarray(terminated).reshape(-1)[0])
                    state["truncated"] = bool(np.asarray(truncated).reshape(-1)[0])
                    if len(state["actions"]) >= max_steps and not state["terminated"]:
                        state["truncated"] = True
                    state["successful"] = state["successful"] or _success(info)
                    if state["terminated"] or state["truncated"]:
                        break
                metadata = state["spec"].get("replan_metadata")
                if metadata is not None:
                    start_step, start_state, stored_images, image_metadata = start
                    executed_tensor = (
                        torch.stack(executed)
                        if executed
                        else torch.empty(0, 7, dtype=plan.dtype)
                    )
                    state["replan_records"].append(
                        ReplanRecord(
                            suite=str(metadata["suite"]),
                            suite_task_id=int(metadata["suite_task_id"]),
                            global_task_index=int(state["spec"]["task_index"]),
                            canonical_task_uid=str(metadata["canonical_task_uid"]),
                            instruction=str(state["spec"]["instruction"]),
                            reset_seed=int(state["spec"]["reset_seed"]),
                            init_state_index=int(state["init_state_index"]),
                            policy_checkpoint=str(metadata["policy_checkpoint"]),
                            policy_checkpoint_sha256=str(metadata["policy_checkpoint_sha256"]),
                            policy_version=str(metadata["policy_version"]),
                            collection_round=int(metadata["collection_round"]),
                            environment_step=start_step,
                            state=start_state,
                            observations=stored_images,
                            observation_metadata=image_metadata,
                            planned_actions=plan.float(),
                            executed_prefix_length=len(executed),
                            executed_actions=executed_tensor,
                            terminated=bool(state["terminated"]),
                            truncated=bool(state["truncated"]),
                            success=bool(state["successful"]),
                            model_revision=str(metadata["model_revision"]),
                            processor_revision=str(metadata["processor_revision"]),
                            dataset_revision=str(metadata["dataset_revision"]),
                        )
                    )
                state["replans"] += 1
            progress.set_postfix(
                active=len(active),
                completed=sum(
                    bool(state["terminated"] or state["truncated"]) for state in states
                ),
                refresh=True,
            )
    finally:
        progress.close()
        for writer in writers:
            if writer is not None:
                writer.close()

    results: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for state in states:
        if len(state["actions"]) >= max_steps and not state["terminated"]:
            state["truncated"] = True
        action_path = (
            torch.stack(state["actions"])
            if state["actions"]
            else torch.empty(0, 7, dtype=torch.float32)
        )
        delta_norm = (
            torch.linalg.vector_norm(action_path[1:] - action_path[:-1], dim=-1)
            if len(action_path) > 1
            else torch.zeros(1)
        )
        model_latencies = state["model_latencies"]
        environment_latencies = state["environment_latencies"]
        metrics = {
            "success": bool(state["successful"]),
            "terminated": bool(state["terminated"]),
            "truncated": bool(state["truncated"]),
            "steps": len(state["actions"]),
            "replans": int(state["replans"]),
            "model_latency_s": model_latencies,
            "environment_latency_s": environment_latencies,
            "mean_model_latency_s": float(np.mean(model_latencies)),
            "p95_model_latency_s": float(np.quantile(model_latencies, 0.95)),
            "mean_environment_latency_s": float(np.mean(environment_latencies)),
            "mean_action_l2": float(
                torch.linalg.vector_norm(action_path, dim=-1).mean()
            ),
            "mean_action_delta_l2": float(delta_norm.mean()),
        }
        results.append(
            (
                metrics,
                {
                    "replans": tuple(state["replan_records"]),
                    "video_path": state["video_path"],
                    "init_state_index": int(state["init_state_index"]),
                },
            )
        )
    return results


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
    print("Starting LIBERO evaluation", flush=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    initialization = tqdm(
        total=3,
        desc="initialize LIBERO evaluation",
        unit="stage",
        dynamic_ncols=True,
    )
    initialization.set_postfix_str("loading policy and processors", refresh=True)
    policy, identity = load_inference_policy(config)
    initialization.update(1)
    initialization.set_postfix_str("loading dataset manifest", refresh=True)
    save_resolved_config(config, output / "resolved_config.yaml")
    evaluation = config["evaluation"]
    manifest = load_episode_manifest(project_path(config["dataset"]["manifest"]))
    initialization.update(1)
    initialization.set_postfix_str("importing LIBERO environment", refresh=True)
    from lerobot.envs.configs import LiberoEnv

    initialization.update(1)
    initialization.close()

    episodes: list[dict[str, Any]] = []
    total_episodes = sum(
        len(suite_spec["task_ids"]) * len(evaluation["reset_seeds"])
        for suite_spec in evaluation["benchmark"]
    )
    episode_progress = tqdm(
        total=total_episodes,
        desc="evaluate LIBERO",
        unit="episode",
        dynamic_ncols=True,
    )
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
        episode_progress.set_postfix_str(f"creating environments for {suite}", refresh=True)
        env_config = LiberoEnv(
            task=suite,
            task_ids=task_ids,
            fps=int(evaluation.get("fps", 10)),
            observation_height=256,
            observation_width=256,
            episode_length=int(evaluation["suite_max_episode_steps"][suite]),
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
                    init_state_index = init_state_index_for_trial(int(reset_seed))
                    metrics, payload = run_episode(
                        env,
                        env_processor,
                        policy,
                        instruction=instruction,
                        reset_seed=int(reset_seed),
                        init_state_index=init_state_index,
                        task_index=dataset_task_index,
                        execution_horizon=int(config["horizons"]["execution"]),
                        max_steps=int(evaluation["suite_max_episode_steps"][suite]),
                        synchronize_cuda=bool(evaluation.get("synchronize_cuda", True)),
                        progress_description=f"{suite}:{task_id} seed={int(reset_seed)}",
                        replan_metadata={
                            "suite": suite,
                            "suite_task_id": task_id,
                            "canonical_task_uid": task_uid,
                            "policy_checkpoint": identity["checkpoint"],
                            "policy_checkpoint_sha256": identity["checkpoint_sha256"],
                            "policy_version": str(identity.get("training_global_step", "hub")),
                            "collection_round": 0,
                            "model_revision": identity["model_revision"],
                            "processor_revision": identity["processor_revision"],
                            "dataset_revision": config["dataset"]["revision"],
                        },
                    )
                    row = {
                        "suite": suite,
                        "task_id": task_id,
                        "dataset_task_index": dataset_task_index,
                        "canonical_task_uid": task_uid,
                        "instruction": instruction,
                        "reset_seed": int(reset_seed),
                        "init_state_index": int(payload["init_state_index"]),
                        **metrics,
                    }
                    episodes.append(row)
                    episode_progress.update(1)
                    episode_progress.set_postfix(
                        suite=suite,
                        task=task_id,
                        seed=int(reset_seed),
                        success=bool(metrics["success"]),
                        refresh=True,
                    )
                    if rollout_store is not None:
                        rollout_store.append(
                            RolloutEpisode(replans=payload["replans"], split="test")
                        )
        finally:
            for group in environments.values():
                for env in group.values():
                    env.close()
    episode_progress.close()
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


def _collect_student_rollouts_serial(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    initialization = tqdm(
        total=3,
        desc="initialize LIBERO rollout collection",
        unit="stage",
        dynamic_ncols=True,
    )
    initialization.set_postfix_str("loading policy and processors", refresh=True)
    policy, identity = load_inference_policy(config)
    initialization.update(1)
    initialization.set_postfix_str("loading dataset manifest", refresh=True)
    collection = config["collection"]
    manifest = load_episode_manifest(project_path(config["dataset"]["manifest"]))
    initialization.update(1)
    initialization.set_postfix_str("importing LIBERO environment", refresh=True)
    from lerobot.envs.configs import LiberoEnv

    initialization.update(1)
    initialization.close()

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
            "seed_protocol": {
                "simulator_seed_field": "reset_seed",
                "init_state_index_field": "init_state_index",
            },
        }
    )
    save_resolved_config(config, output / "resolved_config.yaml")
    validation_task_ids = {int(value) for value in collection["validation_task_ids"]}
    total_episodes = sum(
        len(suite_spec["task_ids"]) * len(collection["train_reset_seeds"])
        + len(validation_task_ids.intersection(int(value) for value in suite_spec["task_ids"]))
        * len(collection["validation_reset_seeds"])
        for suite_spec in collection["benchmark"]
    )
    episode_progress = tqdm(
        total=total_episodes,
        desc="collect LIBERO rollouts",
        unit="episode",
        dynamic_ncols=True,
    )
    for suite_spec in collection["benchmark"]:
        suite = str(suite_spec["suite"])
        task_ids = [int(value) for value in suite_spec["task_ids"]]
        batch_size = int(collection.get("batch_size", 1))
        if batch_size < 1:
            raise ValueError("rollout collection batch_size must be positive")
        episode_progress.set_postfix_str(f"creating environments for {suite}", refresh=True)
        env_config = LiberoEnv(
            task=suite,
            task_ids=task_ids,
            fps=int(collection.get("fps", 10)),
            observation_height=256,
            observation_width=256,
            episode_length=int(collection["suite_max_episode_steps"][suite]),
        )
        env_processor, _ = env_config.get_env_processors()
        environments = env_config.create_envs(n_envs=1, use_async_envs=False)
        try:
            task_metadata: dict[int, tuple[str, int, str]] = {}
            for task_id in task_ids:
                env = environments[suite][task_id]
                instruction = str(env.call("task_description")[0])
                dataset_task_index, task_uid = _canonical_task_identity(
                    manifest, suite, int(task_id), instruction
                )
                task_metadata[task_id] = (instruction, dataset_task_index, task_uid)
            for split, seeds, init_state_indices in (
                (
                    "train",
                    collection["train_reset_seeds"],
                    collection["train_init_state_indices"],
                ),
                (
                    "validation",
                    collection["validation_reset_seeds"],
                    collection["validation_init_state_indices"],
                ),
            ):
                split_task_ids = (
                    task_ids
                    if split == "train"
                    else [task_id for task_id in task_ids if task_id in validation_task_ids]
                )
                for reset_seed, init_state_index in zip(
                    seeds, init_state_indices, strict=True
                ):
                    for chunk_start in range(0, len(split_task_ids), batch_size):
                        chunk = split_task_ids[chunk_start : chunk_start + batch_size]
                        specs: list[dict[str, Any]] = []
                        for task_id in chunk:
                            instruction, dataset_task_index, task_uid = task_metadata[task_id]
                            specs.append(
                                {
                                    "env": environments[suite][task_id],
                                    "instruction": instruction,
                                    "reset_seed": int(reset_seed),
                                    "init_state_index": int(init_state_index),
                                    "task_index": dataset_task_index,
                                    "replan_metadata": {
                                        "suite": suite,
                                        "suite_task_id": task_id,
                                        "canonical_task_uid": task_uid,
                                        "policy_checkpoint": identity["checkpoint"],
                                        "policy_checkpoint_sha256": identity["checkpoint_sha256"],
                                        "policy_version": str(
                                            identity.get("training_global_step", "hub")
                                        ),
                                        "collection_round": int(
                                            collection["collection_round"]
                                        ),
                                        "model_revision": identity["model_revision"],
                                        "processor_revision": identity["processor_revision"],
                                        "dataset_revision": config["dataset"]["revision"],
                                    },
                                    "video_path": (
                                        output
                                        / "validation_videos"
                                        / suite
                                        / (
                                            f"task-{task_id:02d}_seed-{int(reset_seed)}_"
                                            f"init-{int(init_state_index):02d}.mp4"
                                        )
                                        if split == "validation"
                                        and bool(collection["save_validation_videos"])
                                        else None
                                    ),
                                }
                            )
                        results = run_episode_batch(
                            specs,
                            env_processor,
                            policy,
                            execution_horizon=int(config["horizons"]["execution"]),
                            max_steps=int(collection["suite_max_episode_steps"][suite]),
                            synchronize_cuda=True,
                            progress_description=(
                                f"{suite} {split} seed={int(reset_seed)} "
                                f"batch={chunk_start // batch_size + 1}"
                            ),
                            video_fps=int(collection.get("fps", 10)),
                            video_camera=str(collection.get("video_camera", "image")),
                        )
                        for task_id, (_, payload) in zip(chunk, results, strict=True):
                            store.append(
                                RolloutEpisode(replans=payload["replans"], split=split)
                            )
                            episode_progress.update(1)
                            episode_progress.set_postfix(
                                suite=suite,
                                task=task_id,
                                split=split,
                                seed=int(reset_seed),
                                batch=len(chunk),
                                refresh=True,
                            )
        finally:
            for group in environments.values():
                for env in group.values():
                    env.close()
    episode_progress.close()
    report = store.validate()
    report["validation_videos"] = [
        str(path.relative_to(output))
        for path in sorted((output / "validation_videos").rglob("*.mp4"))
    ] if (output / "validation_videos").exists() else []
    (output / "collection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _collect_student_rollouts_multi_gpu(
    config: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Collect disjoint serial task shards in independent GPU processes."""

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"rollout output already exists: {output}")
    collection = config["collection"]
    devices = [str(value) for value in collection["devices"]]
    task_pairs = [
        (str(suite_spec["suite"]), int(task_id))
        for suite_spec in collection["benchmark"]
        for task_id in suite_spec["task_ids"]
    ]
    if not task_pairs:
        raise ValueError("parallel student rollout selection contains no tasks")
    worker_count = min(len(devices), len(task_pairs))
    shards: list[list[tuple[str, int]]] = [[] for _ in range(worker_count)]
    for index, pair in enumerate(task_pairs):
        shards[index % worker_count].append(pair)

    workers_root = output.with_name(output.name + ".parallel-workers")
    if workers_root.exists():
        raise FileExistsError(
            f"stale rollout worker directory exists: {workers_root}; inspect it before retrying"
        )
    workers_root.mkdir(parents=True)
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
        worker_collection = worker_config["collection"]
        worker_collection["parallel_worker"] = True
        worker_collection["devices"] = []
        worker_collection["batch_size"] = 1
        worker_collection["benchmark"] = [
            {
                "suite": suite,
                "task_ids": [task_id for selected_suite, task_id in shard if selected_suite == suite],
            }
            for suite in dict.fromkeys(selected_suite for selected_suite, _ in shard)
        ]
        worker_config["output"]["directory"] = str(worker_output)
        config_path = workers_root / f"{worker_name}.yaml"
        save_resolved_config(worker_config, config_path)
        log_path = workers_root / f"{worker_name}.log"
        log_handle = log_path.open("a", encoding="utf-8")
        environment = dict(os.environ)
        environment.setdefault("MUJOCO_GL", "egl")
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "tmd_policy.cli",
                "rollout",
                "collect-student",
                "--config",
                str(config_path),
                "--output",
                str(worker_output),
            ],
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((worker_index, device, worker_output, process, log_handle))

    validation_task_ids = {int(value) for value in collection["validation_task_ids"]}
    validation_pairs = sum(task_id in validation_task_ids for _, task_id in task_pairs)
    total_episodes = (
        len(task_pairs) * len(collection["train_reset_seeds"])
        + validation_pairs * len(collection["validation_reset_seeds"])
    )
    progress = tqdm(
        total=total_episodes,
        desc=f"student rollout {worker_count}-GPU shards",
        unit="episode",
        dynamic_ncols=True,
    )
    try:
        while any(process.poll() is None for _, _, _, process, _ in processes):
            completed = 0
            for _, _, worker_output, _, _ in processes:
                index_path = worker_output / "episodes.json"
                if index_path.exists():
                    try:
                        completed += len(json.loads(index_path.read_text(encoding="utf-8")))
                    except (OSError, ValueError, json.JSONDecodeError):
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
        raise RuntimeError("parallel student rollout collection failed\n" + "\n".join(details))

    worker_stores = [RolloutStore(worker_output) for _, _, worker_output, _, _ in processes]
    manifests = [
        json.loads((worker.root / "manifest.json").read_text(encoding="utf-8"))
        for worker in worker_stores
    ]
    policy_hashes = {
        str(manifest["policy"]["checkpoint_sha256"]) for manifest in manifests
    }
    if len(policy_hashes) != 1:
        raise RuntimeError("parallel rollout workers used different policy checkpoints")
    metadata = dict(manifests[0])
    metadata.pop("schema", None)
    metadata["parallelism"] = {
        "mode": "independent_serial_processes",
        "devices": devices[:worker_count],
        "workers": worker_count,
        "per_worker_batch_size": 1,
    }
    store = RolloutStore(output)
    store.initialize(metadata)
    save_resolved_config(config, output / "resolved_config.yaml")
    references: list[tuple[tuple[Any, ...], RolloutStore, dict[str, Any]]] = []
    suite_order = {suite: index for index, suite in enumerate(_SUITE_OFFSETS)}
    split_order = {"train": 0, "validation": 1, "test": 2}
    for worker in worker_stores:
        for row in worker.records():
            key = (
                split_order[str(row["split"])],
                suite_order[str(row["suite"])],
                int(row["suite_task_id"]),
                int(row["reset_seed"]),
            )
            references.append((key, worker, row))
    for _, worker, row in sorted(references, key=lambda value: value[0]):
        replans = tuple(
            ReplanRecord(**payload) for payload in worker.load_replans(row)
        )
        store.append(RolloutEpisode(replans=replans, split=str(row["split"])))
    logs = output / "parallel_workers"
    logs.mkdir()
    for worker_index in range(worker_count):
        worker_name = f"worker-{worker_index:02d}"
        shutil.copy2(workers_root / f"{worker_name}.yaml", logs / f"{worker_name}.yaml")
        shutil.copy2(workers_root / f"{worker_name}.log", logs / f"{worker_name}.log")
        worker_videos = workers_root / worker_name / "validation_videos"
        if worker_videos.exists():
            shutil.copytree(
                worker_videos,
                output / "validation_videos",
                dirs_exist_ok=True,
            )
    report = store.validate()
    report["validation_videos"] = [
        str(path.relative_to(output))
        for path in sorted((output / "validation_videos").rglob("*.mp4"))
    ] if (output / "validation_videos").exists() else []
    (output / "collection_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    shutil.rmtree(workers_root)
    return report


def collect_student_rollouts(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    devices = [str(value) for value in config["collection"].get("devices", [])]
    if len(devices) > 1 and not bool(config["collection"].get("parallel_worker", False)):
        return _collect_student_rollouts_multi_gpu(config, output_dir)
    return _collect_student_rollouts_serial(config, output_dir)


__all__ = [
    "collect_student_rollouts",
    "evaluate_libero",
    "run_episode",
    "run_episode_batch",
    "summarize",
    "wilson_interval",
]
