"""Load a trained student checkpoint without constructing training-only teachers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tmd_policy.config import project_path
from tmd_policy.methods.tmd import TMDStage1Program, sample_stage1_generator
from tmd_policy.training.builders import build_student, build_teacher, file_sha256


def _substate(state: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    return {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}


class InferencePolicy(nn.Module):
    def __init__(
        self,
        student: Any,
        *,
        method: str,
        outer_steps: int,
        inner_steps: int,
        student_time_shift_gamma: float,
        tmd_head: nn.Module | None = None,
        sampler_mode: str = "override",
    ) -> None:
        super().__init__()
        self.student = student
        self.method = method
        self.outer_steps = outer_steps
        self.inner_steps = inner_steps
        self.student_time_shift_gamma = student_time_shift_gamma
        self.tmd_head = tmd_head
        self.sampler_mode = sampler_mode

    @property
    def device(self) -> torch.device:
        return self.student.device

    def reset(self) -> None:
        self.student.policy.reset()

    @torch.no_grad()
    def plan(self, canonical_batch: dict[str, Any], *, noise_seed: int) -> Tensor:
        processed = self.student.preprocess_observation(canonical_batch)
        generator = torch.Generator(device=self.student.device).manual_seed(noise_seed)
        batch_size = int(torch.as_tensor(processed["observation.state"]).shape[0])
        noise = torch.randn(
            batch_size,
            50,
            32,
            device=self.student.device,
            dtype=torch.float32,
            generator=generator,
        )
        if self.tmd_head is None and self.sampler_mode == "official":
            normalized = self.student.policy.predict_action_chunk(processed, noise=noise)
            return self.student.postprocessor(normalized)
        condition = self.student.encode_condition(processed)
        if self.tmd_head is None:
            normalized = self.student.sample(
                condition,
                noise,
                self.outer_steps,
                student_time_shift_gamma=self.student_time_shift_gamma,
            )
        else:
            inner = torch.stack(
                [
                    torch.randn(
                        noise.shape,
                        device=noise.device,
                        dtype=noise.dtype,
                        generator=torch.Generator(device=noise.device).manual_seed(
                            noise_seed + 1_000_003 * (index + 1)
                        ),
                    )
                    for index in range(self.outer_steps)
                ]
            )
            normalized = sample_stage1_generator(
                self.student,
                self.tmd_head,
                condition,
                noise,
                outer_steps=self.outer_steps,
                inner_steps=self.inner_steps,
                student_time_shift_gamma=self.student_time_shift_gamma,
                inner_noises=inner,
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

    @torch.no_grad()
    def plan(self, canonical_batch: dict[str, Any], *, noise_seed: int) -> Tensor:
        processed = self.teacher.preprocess_observation(canonical_batch)
        condition = self.teacher.encode_condition(processed)
        generator = torch.Generator(device=self.device).manual_seed(noise_seed)
        noise = torch.randn(
            condition.batch_size,
            50,
            32,
            device=self.device,
            dtype=torch.float32,
            generator=generator,
        )
        normalized = self.teacher.sample(condition, noise, self.num_steps)
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
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = payload["config"]
    if checkpoint_config["method"] != method:
        raise ValueError(f"checkpoint method {checkpoint_config['method']} does not match requested {method}")
    if checkpoint_config["models"] != config["models"] or checkpoint_config["dataset"]["revision"] != config["dataset"]["revision"]:
        raise ValueError("evaluation model/dataset identities differ from the checkpoint")
    state = payload["program"]
    if method == "flow_sft":
        student.configure_trainable(**checkpoint_config["fine_tuning"])
    elif method == "dmd2_flow":
        student.configure_trainable(checkpoint_config["dmd2"]["student_fine_tuning"])
    student.load_state_dict(_substate(state, "student."), strict=True)
    head = None
    if method in {"tmd_stage1", "occupancy_tmd"}:
        stage1 = TMDStage1Program(student, checkpoint_config["tmd"])
        stage1.head.load_state_dict(_substate(state, "head."), strict=True)
        head = stage1.head
    elif method == "tmd_stage2":
        stage1 = TMDStage1Program(student, checkpoint_config["stage1_architecture"])
        stage1.head.load_state_dict(_substate(state, "stage1_head."), strict=True)
        head = stage1.head
    elif method not in {"flow_sft", "dmd2_flow"}:
        raise ValueError(f"unsupported inference checkpoint method: {method}")
    if method == "dmd2_flow":
        sampling_architecture = checkpoint_config["dmd2"]
        outer_steps = int(sampling_architecture["discrete_outer_steps"])
        inner_steps = 1
        student_time_shift_gamma = float(sampling_architecture["student_time_shift_gamma"])
    elif method in {"tmd_stage1", "occupancy_tmd"}:
        sampling_architecture = checkpoint_config["tmd"]
        outer_steps = int(sampling_architecture["discrete_outer_steps"])
        inner_steps = int(sampling_architecture["discrete_inner_steps"])
        student_time_shift_gamma = float(sampling_architecture["student_time_shift_gamma"])
    elif method == "tmd_stage2":
        sampling_architecture = checkpoint_config["stage1_architecture"]
        outer_steps = int(sampling_architecture["discrete_outer_steps"])
        inner_steps = int(sampling_architecture["discrete_inner_steps"])
        student_time_shift_gamma = float(sampling_architecture["student_time_shift_gamma"])
    else:
        outer_steps = int(policy_config["outer_steps"])
        inner_steps = int(policy_config.get("inner_steps", 1))
        student_time_shift_gamma = 1.0
    policy = InferencePolicy(
        student,
        method=method,
        outer_steps=outer_steps,
        inner_steps=inner_steps,
        student_time_shift_gamma=student_time_shift_gamma,
        tmd_head=head,
    ).to(device).eval()
    return policy, {
        "method": method,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": actual_sha,
        "checkpoint_identity_kind": "sha256-of-checkpoint-file",
        "training_global_step": payload["counters"]["global_step"],
        "model_revision": student.model_revision,
        "processor_revision": student.processor_revision,
        "sampling_grid": {
            "outer_steps": outer_steps,
            "inner_steps": inner_steps,
            "student_time_shift_gamma": student_time_shift_gamma,
            "source": "checkpoint architecture" if method != "flow_sft" else "flow-SFT evaluation",
        },
    }


__all__ = ["InferencePolicy", "PI05InferencePolicy", "load_inference_policy"]
