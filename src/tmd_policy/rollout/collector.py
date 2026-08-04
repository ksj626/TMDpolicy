from __future__ import annotations

import hashlib
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tmd_policy.compatibility.actions import CanonicalActionSpace, StateCompatibilityAdapter
from tmd_policy.data.schemas import RolloutChunk


def _canonical_observation(raw: dict[str, Any], env_preprocessor: Any) -> dict[str, Any]:
    from lerobot.envs.utils import preprocess_observation

    return env_preprocessor(preprocess_observation(raw))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@dataclass(frozen=True)
class PlanResult:
    actions: np.ndarray
    outer_noise_seed: int
    inner_noise_seeds: tuple[int, ...]
    preprocessing_latency_s: float
    model_latency_s: float
    postprocessing_latency_s: float


class CanonicalChunkRunner:
    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        *,
        state_adapter: StateCompatibilityAdapter | None = None,
    ) -> None:
        self.policy = policy
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.state_adapter = state_adapter or StateCompatibilityAdapter()
        self.action_space = CanonicalActionSpace()

    def plan(
        self,
        observation: dict[str, Any],
        instruction: str,
        outer_noise_seed: int,
        inner_noise_seeds: tuple[int, ...] | None = None,
    ) -> PlanResult:
        preprocess_started = time.perf_counter()
        model_input = deepcopy(observation)
        model_input["observation.state"] = self.state_adapter.for_student(
            model_input["observation.state"]
        )
        model_input["task"] = [instruction]
        batch = self.preprocessor(model_input)
        preprocessing_latency = time.perf_counter() - preprocess_started
        state = batch["observation.state"]
        device = state.device
        batch_size = state.shape[0]
        chunk_size = int(self.policy.config.chunk_size)
        action_dim = int(self.policy.config.max_action_dim)
        outer_steps = int(self.policy.generator.outer_steps)
        uses_inner_noise = bool(getattr(self.policy, "uses_inner_noise", True))
        if not uses_inner_noise:
            inner_noise_seeds = ()
        elif inner_noise_seeds is None:
            inner_noise_seeds = tuple(
                outer_noise_seed + 1_000_003 * (index + 1) for index in range(outer_steps)
            )
        if uses_inner_noise and len(inner_noise_seeds) != outer_steps:
            raise ValueError(f"expected {outer_steps} inner noise seeds, got {len(inner_noise_seeds)}")
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.inference_mode(), torch.random.fork_rng(devices=devices):
            outer_generator = torch.Generator(device=device).manual_seed(outer_noise_seed)
            shape = (batch_size, chunk_size, action_dim)
            outer_noise = torch.randn(shape, device=device, dtype=torch.float32, generator=outer_generator)
            inner_noises = (
                torch.stack(
                    [
                        torch.randn(
                            shape,
                            device=device,
                            dtype=torch.float32,
                            generator=torch.Generator(device=device).manual_seed(seed),
                        )
                        for seed in inner_noise_seeds
                    ]
                )
                if uses_inner_noise
                else None
            )
            _synchronize(device)
            model_started = time.perf_counter()
            internal = self.policy.predict_action_chunk(
                batch, noise=outer_noise, inner_noises=inner_noises
            )
            _synchronize(device)
            model_latency = time.perf_counter() - model_started
            postprocess_started = time.perf_counter()
            canonical = self.postprocessor(internal)
            _synchronize(device)
            postprocessing_latency = time.perf_counter() - postprocess_started
        actions = self.action_space.project(torch.as_tensor(canonical).detach().cpu()[0])
        self.action_space.validate(actions)
        return PlanResult(
            actions=actions.numpy().astype(np.float32),
            outer_noise_seed=outer_noise_seed,
            inner_noise_seeds=inner_noise_seeds,
            preprocessing_latency_s=preprocessing_latency,
            model_latency_s=model_latency,
            postprocessing_latency_s=postprocessing_latency,
        )


def collect_rollout_episode(
    env: Any,
    env_preprocessor: Any,
    runner: CanonicalChunkRunner,
    *,
    policy_checkpoint: str,
    policy_version: str,
    collection_round: int,
    task_index: int,
    instruction: str,
    reset_seed: int,
    base_noise_seed: int,
    prediction_horizon: int = 50,
    execution_horizon: int = 10,
    max_chunks: int | None = None,
    max_environment_steps: int | None = None,
) -> tuple[list[RolloutChunk], dict[str, float]]:
    if max_environment_steps is not None and max_environment_steps < 1:
        raise ValueError("max_environment_steps must be positive when provided")
    raw, info = env.reset(seed=[reset_seed])
    canonical = _canonical_observation(raw, env_preprocessor)
    records: list[RolloutChunk] = []
    model_latencies: list[float] = []
    preprocessing_latencies: list[float] = []
    postprocessing_latencies: list[float] = []
    environment_latencies: list[float] = []
    done = False
    chunk_index = 0
    episode_success = False
    environment_steps = 0
    local_time_limit_reached = False
    while not done and (max_chunks is None or chunk_index < max_chunks):
        start_observation = deepcopy(canonical)
        outer_noise_seed = base_noise_seed + chunk_index
        plan_result = runner.plan(start_observation, instruction, outer_noise_seed)
        plan = plan_result.actions
        if plan.shape[0] != prediction_horizon:
            raise ValueError(f"policy produced {plan.shape[0]} actions, expected {prediction_horizon}")
        model_latencies.append(plan_result.model_latency_s)
        preprocessing_latencies.append(plan_result.preprocessing_latency_s)
        postprocessing_latencies.append(plan_result.postprocessing_latency_s)
        states = [start_observation["observation.state"][0].detach().cpu().numpy()]
        executed: list[np.ndarray] = []
        terminated_flag = False
        truncated_flag = False
        environment_truncated_flag = False
        chunk_local_time_limit = False
        success = False
        environment_latency = 0.0
        for action in plan[:execution_horizon]:
            environment_started = time.perf_counter()
            raw, _, terminated, truncated, info = env.step(action[None])
            environment_latency += time.perf_counter() - environment_started
            environment_steps += 1
            executed.append(action)
            canonical = _canonical_observation(raw, env_preprocessor)
            states.append(canonical["observation.state"][0].detach().cpu().numpy())
            terminated_flag = bool(np.asarray(terminated).reshape(-1)[0])
            environment_truncated_flag = bool(np.asarray(truncated).reshape(-1)[0])
            truncated_flag = environment_truncated_flag
            if (
                max_environment_steps is not None
                and environment_steps >= max_environment_steps
                and not terminated_flag
                and not truncated_flag
            ):
                truncated_flag = True
                chunk_local_time_limit = True
                local_time_limit_reached = True
            if "is_success" in info:
                success = success or bool(np.asarray(info["is_success"]).reshape(-1)[0])
            if "final_info" in info and info["final_info"] is not None:
                final = np.asarray(info["final_info"], dtype=object).reshape(-1)[0]
                if isinstance(final, dict):
                    success = success or bool(final.get("is_success", False))
            done = terminated_flag or truncated_flag
            if done:
                break
        episode_success = episode_success or success
        environment_latencies.append(environment_latency)
        observation_key = hashlib.sha256(
            f"{policy_version}:{collection_round}:{reset_seed}:{chunk_index}".encode()
        ).hexdigest()[:24]
        images = {
            key: value[0].detach().cpu().numpy()
            for key, value in start_observation.items()
            if key.startswith("observation.images.")
        }
        executed_array = np.asarray(executed, dtype=np.float32)
        records.append(
            RolloutChunk(
                sample_id=f"rollout-r{collection_round}-{policy_version}-{reset_seed}-{chunk_index}",
                observation_id=observation_key,
                policy_checkpoint=policy_checkpoint,
                policy_version=policy_version,
                collection_round=collection_round,
                task_index=task_index,
                instruction=instruction,
                chunk_index=chunk_index,
                plan_actions=plan,
                executed_actions=executed_array,
                path_states=np.asarray(states, dtype=np.float32),
                path_valid=np.ones(len(executed), dtype=bool),
                success=success,
                terminated=terminated_flag,
                truncated=truncated_flag,
                environment_truncated=environment_truncated_flag,
                local_time_limit=chunk_local_time_limit,
                reset_seed=reset_seed,
                outer_noise_seed=outer_noise_seed,
                inner_noise_seeds=plan_result.inner_noise_seeds,
                preprocessing_latency_s=plan_result.preprocessing_latency_s,
                model_latency_s=plan_result.model_latency_s,
                postprocessing_latency_s=plan_result.postprocessing_latency_s,
                environment_latency_s=environment_latency,
                chunk_start_images=images,
            )
        )
        chunk_index += 1
    return records, {
        "success": float(episode_success),
        "chunks": float(len(records)),
        "mean_model_latency_s": (
            float(np.mean(model_latencies)) if model_latencies else float("nan")
        ),
        "total_model_latency_s": float(np.sum(model_latencies)),
        "total_preprocessing_latency_s": float(np.sum(preprocessing_latencies)),
        "total_postprocessing_latency_s": float(np.sum(postprocessing_latencies)),
        "total_environment_latency_s": float(np.sum(environment_latencies)),
        "environment_steps": float(environment_steps),
        "local_time_limit_reached": float(local_time_limit_reached),
    }
