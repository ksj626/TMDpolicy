"""Load a trained student checkpoint without constructing training-only teachers."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor, nn

from tmd_policy.backends.lerobot.smolvla_student import resolve_trainable_state_keys
from tmd_policy.config import project_path
from tmd_policy.training.builders import build_student, build_teacher, file_sha256


def _substate(state: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    return {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}


def _seeded_noise(
    seeds: int | Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
    shape: tuple[int, ...],
    offset: int = 0,
) -> Tensor:
    normalized = [int(seeds)] if isinstance(seeds, int) else [int(seed) for seed in seeds]
    if len(normalized) == 1 and batch_size > 1:
        # Preserve the historical one-generator stream for callers that pass a
        # scalar seed with a batch. Episode batching passes one seed per item.
        generator = torch.Generator(device=device).manual_seed(normalized[0] + offset)
        return torch.randn(
            batch_size,
            *shape,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    if len(normalized) != batch_size:
        raise ValueError(f"received {len(normalized)} noise seeds for batch size {batch_size}")
    return torch.stack(
        [
            torch.randn(
                *shape,
                device=device,
                dtype=torch.float32,
                generator=torch.Generator(device=device).manual_seed(seed + offset),
            )
            for seed in normalized
        ]
    )


def _dmd2_seeded_noises(
    seeds: int | Sequence[int],
    *,
    batch_size: int,
    device: torch.device,
    outer_steps: int,
) -> tuple[Tensor, torch.Generator | None, Tensor | None]:
    if isinstance(seeds, int):
        generator = torch.Generator(device=device).manual_seed(seeds)
        noise = torch.randn(
            batch_size, 50, 32, device=device, dtype=torch.float32, generator=generator
        )
        return noise, generator, None
    normalized = [int(seed) for seed in seeds]
    if len(normalized) != batch_size:
        raise ValueError(f"received {len(normalized)} noise seeds for batch size {batch_size}")
    initial: list[Tensor] = []
    renoise_per_episode: list[Tensor] = []
    for seed in normalized:
        generator = torch.Generator(device=device).manual_seed(seed)
        initial.append(
            torch.randn(50, 32, device=device, dtype=torch.float32, generator=generator)
        )
        renoise_per_episode.append(
            torch.stack(
                [
                    torch.randn(50, 32, device=device, dtype=torch.float32, generator=generator)
                    for _ in range(max(0, outer_steps - 1))
                ]
            )
            if outer_steps > 1
            else torch.empty(0, 50, 32, device=device, dtype=torch.float32)
        )
    renoise = torch.stack(renoise_per_episode, dim=1)
    return torch.stack(initial), None, renoise


def _load_student_checkpoint_state(
    student: Any,
    state: dict[str, Tensor],
    *,
    checkpoint_format: str,
    trainable_parameter_names: list[str] | tuple[str, ...],
) -> None:
    student_state = _substate(state, "student.")
    if checkpoint_format == "tmdpolicy.inference/v1":
        expected_delta = set(resolve_trainable_state_keys(student))
        if set(student_state) != expected_delta:
            raise ValueError(
                "inference checkpoint student delta does not match the checkpoint's "
                f"fine-tuning contract: {sorted(student_state)} != {sorted(expected_delta)}"
            )
        stored_names = set(trainable_parameter_names)
        expected_stored_names = {f"student.{name}" for name in expected_delta}
        if stored_names != expected_stored_names:
            raise ValueError("inference checkpoint trainable-parameter manifest is inconsistent")
        incompatible = student.load_state_dict(student_state, strict=False)
        if incompatible.unexpected_keys or expected_delta.intersection(incompatible.missing_keys):
            raise RuntimeError(
                "failed to apply the complete trained student delta: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
    else:
        student.load_state_dict(student_state, strict=True)


class InferencePolicy(nn.Module):
    def __init__(
        self,
        student: Any,
        *,
        method: str,
        outer_steps: int,
        inner_steps: int,
        student_time_shift_gamma: float,
        sampler_mode: str = "override",
    ) -> None:
        super().__init__()
        self.student = student
        self.method = method
        self.outer_steps = outer_steps
        self.inner_steps = inner_steps
        self.student_time_shift_gamma = student_time_shift_gamma
        self.sampler_mode = sampler_mode

    @property
    def device(self) -> torch.device:
        return self.student.device

    def reset(self) -> None:
        self.student.policy.reset()

    @property
    def inference_work_units(self) -> int:
        return self.outer_steps

    @torch.no_grad()
    def plan(
        self,
        canonical_batch: dict[str, Any],
        *,
        noise_seed: int | Sequence[int],
        step_callback: Callable[[], None] | None = None,
    ) -> Tensor:
        processed = self.student.preprocess_observation(canonical_batch)
        batch_size = int(torch.as_tensor(processed["observation.state"]).shape[0])
        if self.method == "dmd2_flow":
            noise, generator, renoise_noises = _dmd2_seeded_noises(
                noise_seed,
                batch_size=batch_size,
                device=self.student.device,
                outer_steps=self.outer_steps,
            )
        else:
            noise = _seeded_noise(
                noise_seed,
                batch_size=batch_size,
                device=self.student.device,
                shape=(50, 32),
            )
            generator = None
            renoise_noises = None
        if self.sampler_mode == "official":
            handle = None
            if step_callback is not None:
                handle = self.student.flow.action_out_proj.register_forward_hook(
                    lambda _module, _inputs, _output: step_callback()
                )
            try:
                normalized = self.student.policy.predict_action_chunk(processed, noise=noise)
            finally:
                if handle is not None:
                    handle.remove()
            return self.student.postprocessor(normalized)
        condition = self.student.encode_condition(processed)
        if self.method == "dmd2_flow":
            normalized = self.student.sample_denoise_renoise(
                condition,
                noise,
                self.outer_steps,
                student_time_shift_gamma=self.student_time_shift_gamma,
                generator=generator,
                renoise_noises=renoise_noises,
                step_callback=step_callback,
            )
        else:
            normalized = self.student.sample(
                condition,
                noise,
                self.outer_steps,
                student_time_shift_gamma=self.student_time_shift_gamma,
                step_callback=step_callback,
            )
        return self.student.postprocessor(normalized[..., :7])


class PI05InferencePolicy(nn.Module):
    """Direct deterministic official PI0.5 Euler evaluation policy."""

    def __init__(self, teacher: Any, *, num_steps: int) -> None:
        super().__init__()
        if num_steps < 1:
            raise ValueError("PI0.5 evaluation steps must be positive")
        self.teacher = teacher
        self.num_steps = num_steps

    @property
    def device(self) -> torch.device:
        return self.teacher.device

    def reset(self) -> None:
        if hasattr(self.teacher.policy, "reset"):
            self.teacher.policy.reset()

    @property
    def inference_work_units(self) -> int:
        return self.num_steps

    @torch.no_grad()
    def plan(
        self,
        canonical_batch: dict[str, Any],
        *,
        noise_seed: int | Sequence[int],
        step_callback: Callable[[], None] | None = None,
    ) -> Tensor:
        processed = self.teacher.preprocess_observation(canonical_batch)
        condition = self.teacher.encode_condition(processed)
        noise = _seeded_noise(
            noise_seed,
            batch_size=condition.batch_size,
            device=self.device,
            shape=(50, 32),
        )
        normalized = self.teacher.sample(
            condition,
            noise,
            self.num_steps,
            step_callback=step_callback,
        )
        canonical = self.teacher.postprocess_action(normalized)
        if canonical.shape != (condition.batch_size, 50, 7):
            raise RuntimeError(f"PI0.5 postprocessor returned {tuple(canonical.shape)}, expected [B,50,7]")
        return canonical


def load_inference_policy(
    config: dict[str, Any],
) -> tuple[InferencePolicy | PI05InferencePolicy, dict[str, Any]]:
    policy_config = config["policy"]
    method = str(policy_config["method"])
    device = str(policy_config["device"])
    if method == "pi05":
        teacher = build_teacher(config, device=device)
        steps = int(policy_config.get("num_steps", teacher.policy.config.num_inference_steps))
        locator = f"hf://{teacher.model_id}@{teacher.model_revision}"
        return PI05InferencePolicy(teacher, num_steps=steps).eval(), {
            "method": method,
            "checkpoint": locator,
            "checkpoint_sha256": hashlib.sha256(locator.encode("utf-8")).hexdigest(),
            "checkpoint_identity_kind": "sha256-of-immutable-hub-locator",
            "model_revision": teacher.model_revision,
            "processor_revision": teacher.processor_revision,
            "num_steps": steps,
            "seed_rule": "100000007*global_task_index + 10007*reset_seed + replan_index",
            "coordinate_contract": "PI0.5 [B,50,32] internal -> official postprocessor [B,50,7]",
        }
    student = build_student(config, device=device)
    if method == "smolvla":
        mode = str(policy_config.get("sampler_mode", "official"))
        if mode not in {"official", "override"}:
            raise ValueError("SmolVLA sampler_mode must be official or override")
        checkpoint_steps = int(student.policy.config.num_steps)
        if mode == "official":
            if checkpoint_steps != 10:
                raise RuntimeError(
                    f"official SmolVLA baseline requires checkpoint num_steps==10, got {checkpoint_steps}"
                )
            if "num_steps" in policy_config or "outer_steps" in policy_config:
                raise ValueError("official SmolVLA mode forbids step overrides")
            steps = checkpoint_steps
            classification = "official checkpoint sampler"
        else:
            steps = int(policy_config["num_steps"])
            classification = str(policy_config["classification"])
            if not classification or "ablation" not in classification.lower():
                raise ValueError("SmolVLA override mode must be explicitly classified as an ablation")
        locator = f"hf://{student.model_id}@{student.model_revision}"
        return (
            InferencePolicy(
                student,
                method=method,
                outer_steps=steps,
                inner_steps=1,
                student_time_shift_gamma=1.0,
                sampler_mode=mode,
            ).eval(),
            {
                "method": method,
                "checkpoint": locator,
                "checkpoint_sha256": hashlib.sha256(locator.encode("utf-8")).hexdigest(),
                "checkpoint_identity_kind": "sha256-of-immutable-hub-locator",
                "model_revision": student.model_revision,
                "processor_revision": student.processor_revision,
                "num_steps": steps,
                "sampler_mode": mode,
                "classification": classification,
            },
        )
    checkpoint = project_path(policy_config["checkpoint"])
    actual_sha = file_sha256(checkpoint)
    expected_sha = policy_config["checkpoint_sha256"]
    if str(expected_sha).lower() == "auto":
        policy_config["checkpoint_sha256"] = actual_sha
        expected_sha = actual_sha
    if actual_sha != expected_sha:
        raise RuntimeError(f"policy checkpoint SHA-256 mismatch: {actual_sha} != {expected_sha}")
    # Full resume checkpoints contain large fake-score/optimizer tensors that
    # inference never reads. Memory-map their storages so evaluating an
    # intermediate checkpoint does not materialize the whole training state in
    # host RAM. Inference-delta checkpoints benefit as well.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    checkpoint_format = str(payload.get("format", ""))
    if checkpoint_format not in {"tmdpolicy.training/v1", "tmdpolicy.inference/v1"}:
        raise ValueError(f"unsupported inference checkpoint format: {checkpoint_format!r}")
    checkpoint_config = payload["config"]
    if checkpoint_config["method"] != method:
        raise ValueError(f"checkpoint method {checkpoint_config['method']} does not match requested {method}")
    if checkpoint_config["models"] != config["models"] or checkpoint_config["dataset"]["revision"] != config["dataset"]["revision"]:
        raise ValueError("evaluation model/dataset identities differ from the checkpoint")
    state = payload["program"]
    if method != "dmd2_flow":
        raise ValueError(f"unsupported inference checkpoint method: {method}")
    student.configure_trainable(checkpoint_config["dmd2"]["student_fine_tuning"])
    _load_student_checkpoint_state(
        student,
        state,
        checkpoint_format=checkpoint_format,
        trainable_parameter_names=payload.get("trainable_parameter_names", ()),
    )
    sampling_architecture = checkpoint_config["dmd2"]
    checkpoint_outer_steps = int(sampling_architecture["discrete_outer_steps"])
    outer_steps = int(policy_config.get("outer_steps", checkpoint_outer_steps))
    if outer_steps < 1:
        raise ValueError("DMD2 evaluation outer_steps must be positive")
    inner_steps = 1
    student_time_shift_gamma = float(sampling_architecture["student_time_shift_gamma"])
    policy = InferencePolicy(
        student,
        method=method,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        student_time_shift_gamma=student_time_shift_gamma,
    ).to(device).eval()
    return policy, {
        "method": method,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_sha,
        "checkpoint_identity_kind": "sha256-of-checkpoint-file",
        "checkpoint_format": checkpoint_format,
        "training_global_step": payload["counters"]["global_step"],
        "model_revision": student.model_revision,
        "processor_revision": student.processor_revision,
        "sampling_grid": {
            "outer_steps": outer_steps,
            "inner_steps": inner_steps,
            "student_time_shift_gamma": student_time_shift_gamma,
            "source": "DMD2 evaluation ablation override" if "outer_steps" in policy_config else "checkpoint architecture",
            "sampler": "DMD2 denoise-renoise",
        },
    }


__all__ = ["InferencePolicy", "PI05InferencePolicy", "load_inference_policy"]
