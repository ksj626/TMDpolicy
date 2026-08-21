"""Single-episode, real-environment audit of the tensors entering SmolVLA."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor
from tqdm.auto import tqdm

from tmd_policy.config import save_resolved_config
from tmd_policy.libero_protocol import init_state_index_for_trial, set_fixed_init_state_index

from .libero import _canonical_observation, _success
from .policy import InferencePolicy, load_inference_policy


SUITE_OFFSETS = {
    "libero_spatial": 0,
    "libero_object": 10,
    "libero_goal": 20,
    "libero_10": 30,
}


def tensor_summary(value: Any) -> dict[str, Any]:
    """Return bounded, JSON-safe diagnostics without retaining the tensor."""

    tensor = torch.as_tensor(value).detach()
    result: dict[str, Any] = {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": tensor.numel(),
    }
    if tensor.numel() == 0:
        return result
    if tensor.dtype == torch.bool:
        result["true_fraction"] = float(tensor.float().mean().cpu())
        return result
    if tensor.is_floating_point():
        finite = torch.isfinite(tensor)
        result["finite_fraction"] = float(finite.float().mean().cpu())
        if bool(finite.any()):
            values = tensor[finite].float()
            result.update(
                {
                    "min": float(values.min().cpu()),
                    "max": float(values.max().cpu()),
                    "mean": float(values.mean().cpu()),
                    "std": float(values.std(unbiased=False).cpu()),
                    "l2": float(torch.linalg.vector_norm(values).cpu()),
                }
            )
        return result
    result["min"] = int(tensor.min().cpu())
    result["max"] = int(tensor.max().cpu())
    flat = tensor.flatten()
    result["first_values"] = [int(item) for item in flat[:16].cpu()]
    return result


def _summarize_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key, item in sorted(value.items()):
        if isinstance(item, Mapping):
            summary[str(key)] = _summarize_mapping(item)
        elif isinstance(item, (Tensor, np.ndarray, list, tuple)) and not (
            isinstance(item, (list, tuple)) and item and isinstance(item[0], str)
        ):
            try:
                summary[str(key)] = tensor_summary(item)
            except (TypeError, ValueError):
                summary[str(key)] = {"type": type(item).__name__}
        elif isinstance(item, (str, int, float, bool)) or item is None:
            summary[str(key)] = item
        else:
            summary[str(key)] = {"type": type(item).__name__}
    return summary


def _save_prepared_image(image: Tensor, path: Path) -> None:
    """Save the exact post-resize SmolVLA image after reversing [-1,1] scaling."""

    import imageio.v2 as imageio

    value = image.detach().float().cpu()
    if value.ndim == 4:
        if value.shape[0] != 1:
            raise ValueError(f"debug image batch must be one, got {tuple(value.shape)}")
        value = value[0]
    if value.ndim != 3 or value.shape[0] != 3:
        raise ValueError(f"prepared image must be CHW RGB, got {tuple(value.shape)}")
    pixels = (((value.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, pixels)


def _student_backend(policy: InferencePolicy) -> Any:
    student = getattr(policy, "student", None)
    if student is None:
        raise ValueError(
            "input audit currently supports SmolVLA-backed policies "
            "(smolvla or dmd2_flow), not PI0.5"
        )
    return student


def _validate_model_inputs(
    student: Any,
    processed: Mapping[str, Any],
    *,
    replan_index: int,
    images_dir: Path | None,
) -> dict[str, Any]:
    config = student.policy.config
    expected_image_keys = [str(key) for key in config.image_features]
    present_image_keys = [key for key in expected_image_keys if key in processed]
    missing_image_keys = [key for key in expected_image_keys if key not in processed]
    prepared_images, image_masks = student.policy.prepare_images(dict(processed))
    prepared_state = student.policy.prepare_state(dict(processed))
    state_before_padding = torch.as_tensor(processed["observation.state"])
    language_tensors = {
        key: torch.as_tensor(value)
        for key, value in sorted(processed.items())
        if "language" in key
    }

    checks = {
        "two_real_cameras": len(present_image_keys) == 2,
        "prepared_image_count_matches_present": len(prepared_images) == len(present_image_keys),
        "camera3_not_materialized": "observation.images.camera3" not in processed,
        "state_is_8d_before_padding": list(state_before_padding.shape) == [1, 8],
        "state_is_32d_after_padding": list(prepared_state.shape) == [1, 32],
        "all_prepared_images_finite": all(bool(torch.isfinite(image).all()) for image in prepared_images),
        "all_prepared_images_valid": all(
            image.ndim == 4 and image.shape[0] == 1 and image.shape[1] == 3
            for image in prepared_images
        ),
        "all_image_masks_true": all(bool(mask.all()) for mask in image_masks),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"model input audit failed at replan {replan_index}: {failed}")

    saved_images: list[str] = []
    if images_dir is not None:
        for index, image in enumerate(prepared_images, start=1):
            path = images_dir / f"replan-{replan_index:04d}-camera-{index}.png"
            _save_prepared_image(image, path)
            saved_images.append(str(path))

    return {
        "checkpoint_feature_schema": {
            "expected_image_keys": expected_image_keys,
            "present_image_keys": present_image_keys,
            "missing_image_keys": missing_image_keys,
            "empty_cameras": int(config.empty_cameras),
            "declared_state_shape": list(config.input_features["observation.state"].shape),
        },
        "actual_model_inputs": {
            "image_count": len(prepared_images),
            "images": [tensor_summary(image) for image in prepared_images],
            "image_channels": [
                [tensor_summary(channel) for channel in image[0]] for image in prepared_images
            ],
            "image_masks": [tensor_summary(mask) for mask in image_masks],
            "state_before_padding": tensor_summary(state_before_padding),
            "state_before_padding_values": state_before_padding.detach().float().cpu().tolist(),
            "state_after_padding": tensor_summary(prepared_state),
            "state_after_padding_values": prepared_state.detach().float().cpu().tolist(),
            "language_tensors": {
                key: {
                    "summary": tensor_summary(value),
                    "values": value.detach().cpu().tolist(),
                }
                for key, value in language_tensors.items()
            },
        },
        "checks": checks,
        "saved_images": saved_images,
    }


def _print_replan(record: Mapping[str, Any]) -> None:
    actual = record["model_input_audit"]["actual_model_inputs"]
    schema = record["model_input_audit"]["checkpoint_feature_schema"]
    print(
        f"\n[replan {record['replan_index']:04d} | env step {record['environment_step']:04d}]",
        flush=True,
    )
    print(f"  canonical keys: {record['canonical_keys']}", flush=True)
    print(f"  processed keys: {record['processed_keys']}", flush=True)
    print(
        "  cameras: "
        f"declared={schema['expected_image_keys']} present={schema['present_image_keys']} "
        f"missing={schema['missing_image_keys']} actual_count={actual['image_count']}",
        flush=True,
    )
    for index, image in enumerate(actual["images"], start=1):
        print(
            f"    camera[{index}]: shape={image['shape']} dtype={image['dtype']} "
            f"range=[{image['min']:.4g},{image['max']:.4g}] "
            f"mean={image['mean']:.4g} std={image['std']:.4g}",
            flush=True,
        )
    before = actual["state_before_padding"]
    after = actual["state_after_padding"]
    print(
        f"  state: before={before['shape']} after={after['shape']} "
        f"finite={before['finite_fraction']:.3f}",
        flush=True,
    )
    print(
        f"    normalized values: {actual['state_before_padding_values'][0]}",
        flush=True,
    )
    print(
        f"  plan: shape={record['plan']['shape']} range="
        f"[{record['plan']['min']:.4g},{record['plan']['max']:.4g}] "
        f"inference={record['inference_seconds']:.3f}s",
        flush=True,
    )


def audit_libero_episode(
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    suite: str,
    task_id: int,
    reset_seed: int,
    max_episode_steps: int | None,
    execution_horizon: int | None,
    save_images: bool,
) -> dict[str, Any]:
    """Run one real episode and audit every SmolVLA input at every replan."""

    if suite not in SUITE_OFFSETS:
        raise ValueError(f"unsupported LIBERO suite: {suite}")
    if not 0 <= task_id <= 9:
        raise ValueError("LIBERO task ID must be in [0,9]")
    if reset_seed < 0:
        raise ValueError("reset seed must be nonnegative")

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite debug output: {output}")
    output.mkdir(parents=True)
    save_resolved_config(config, output / "resolved_config.yaml")

    evaluation = config["evaluation"]
    configured_steps = int(evaluation["suite_max_episode_steps"][suite])
    max_steps = configured_steps if max_episode_steps is None else int(max_episode_steps)
    execute = int(config["horizons"]["execution"] if execution_horizon is None else execution_horizon)
    if max_steps < 1 or execute < 1 or execute > 50:
        raise ValueError("max episode steps must be positive and execution horizon must be in [1,50]")

    policy, identity = load_inference_policy(config)
    if not isinstance(policy, InferencePolicy):
        raise ValueError("input audit currently requires a SmolVLA-backed policy")
    student = _student_backend(policy)

    from lerobot.envs.configs import LiberoEnv
    from lerobot.envs.utils import NEW_ROLLOUT_OPTION

    env_config = LiberoEnv(
        task=suite,
        task_ids=[task_id],
        fps=int(evaluation.get("fps", 20)),
        observation_height=256,
        observation_width=256,
        episode_length=max_steps,
        control_mode=str(evaluation.get("control_mode", "relative")),
        hard_reset=bool(evaluation.get("hard_reset", True)),
    )
    env_processor, _ = env_config.get_env_processors()
    environments = env_config.create_envs(n_envs=1, use_async_envs=False)
    env = environments[suite][task_id]
    records_path = output / "model_inputs.jsonl"
    images_dir = output / "model_input_images" if save_images else None
    records: list[dict[str, Any]] = []
    actions: list[Tensor] = []
    replans = 0
    successful = False
    terminated_value = False
    truncated_value = False
    started = time.perf_counter()
    try:
        policy.reset()
        init_state_index = init_state_index_for_trial(reset_seed)
        resolved_init_state = set_fixed_init_state_index(env, init_state_index)
        raw, info = env.reset(seed=[reset_seed], options={NEW_ROLLOUT_OPTION: True})
        instruction = str(env.call("task_description")[0])
        successful = _success(info)
        progress = tqdm(total=max_steps, desc="audit real LIBERO episode", unit="env-step", dynamic_ncols=True)
        try:
            while len(actions) < max_steps and not (terminated_value or truncated_value):
                canonical = _canonical_observation(raw, env_processor)
                model_batch = dict(canonical)
                model_batch["task"] = [instruction]
                processed = student.preprocess_observation(dict(model_batch))
                model_audit = _validate_model_inputs(
                    student,
                    processed,
                    replan_index=replans,
                    images_dir=images_dir,
                )
                noise_seed = (
                    100_000_007 * (SUITE_OFFSETS[suite] + task_id)
                    + 10_007 * reset_seed
                    + replans
                )
                inference_started = time.perf_counter()
                plan = policy.plan(model_batch, noise_seed=noise_seed)[0].detach().cpu()
                inference_seconds = time.perf_counter() - inference_started
                plan_summary = tensor_summary(plan)
                if plan.shape != (50, 7) or not math.isclose(plan_summary["finite_fraction"], 1.0):
                    raise RuntimeError(f"invalid action plan at replan {replans}: {plan_summary}")
                record = {
                    "replan_index": replans,
                    "environment_step": len(actions),
                    "noise_seed": noise_seed,
                    "instruction": instruction,
                    "raw_observation": _summarize_mapping(raw),
                    "canonical_keys": sorted(str(key) for key in canonical),
                    "canonical_observation": _summarize_mapping(canonical),
                    "processed_keys": sorted(str(key) for key in processed),
                    "processed_observation": _summarize_mapping(processed),
                    "model_input_audit": model_audit,
                    "plan": plan_summary,
                    "plan_values": plan.float().tolist(),
                    "inference_seconds": inference_seconds,
                }
                records.append(record)
                with records_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                _print_replan(record)

                for action in plan[:execute]:
                    if len(actions) >= max_steps:
                        truncated_value = True
                        break
                    raw, _, terminated, truncated, info = env.step(action.numpy()[None])
                    actions.append(action.float())
                    progress.update(1)
                    terminated_value = bool(np.asarray(terminated).reshape(-1)[0])
                    truncated_value = bool(np.asarray(truncated).reshape(-1)[0])
                    successful = successful or _success(info)
                    if terminated_value or truncated_value:
                        break
                replans += 1
                progress.set_postfix(replans=replans, success=successful, refresh=True)
        finally:
            progress.close()
    finally:
        env.close()

    report = {
        "purpose": "real single-episode SmolVLA model-input audit",
        "policy": identity,
        "suite": suite,
        "task_id": task_id,
        "instruction": instruction,
        "reset_seed": reset_seed,
        "init_state_index": resolved_init_state,
        "max_episode_steps": max_steps,
        "execution_horizon": execute,
        "steps": len(actions),
        "replans": replans,
        "success": successful,
        "terminated": terminated_value,
        "truncated": truncated_value,
        "elapsed_s": time.perf_counter() - started,
        "all_checks_passed": all(
            all(record["model_input_audit"]["checks"].values()) for record in records
        ),
        "records": str(records_path),
        "images": str(images_dir) if images_dir is not None else None,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["audit_libero_episode", "tensor_summary"]
