"""Load a trained student checkpoint without constructing training-only teachers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from tmd_policy.config import project_path
from tmd_policy.methods.tmd import TMDStage1Program, sample_stage1_generator
from tmd_policy.training.builders import build_student, file_sha256


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
        tmd_head: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.student = student
        self.method = method
        self.outer_steps = outer_steps
        self.inner_steps = inner_steps
        self.tmd_head = tmd_head

    def reset(self) -> None:
        self.student.policy.reset()

    @torch.no_grad()
    def plan(self, canonical_batch: dict[str, Any], *, noise_seed: int) -> Tensor:
        processed = self.student.preprocess_observation(canonical_batch)
        condition = self.student.encode_condition(processed)
        generator = torch.Generator(device=self.student.device).manual_seed(noise_seed)
        noise = torch.randn(
            condition.batch_size,
            50,
            32,
            device=self.student.device,
            dtype=torch.float32,
            generator=generator,
        )
        if self.tmd_head is None:
            normalized = self.student.sample(condition, noise, self.outer_steps)
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
                inner_noises=inner,
            )
        return self.student.postprocessor(normalized[..., :7])


def load_inference_policy(config: dict[str, Any]) -> tuple[InferencePolicy, dict[str, Any]]:
    policy_config = config["policy"]
    method = str(policy_config["method"])
    device = str(policy_config["device"])
    student = build_student(config, device=device)
    if method == "smolvla":
        locator = f"hf://{student.model_id}@{student.model_revision}"
        return (
            InferencePolicy(
                student,
                method=method,
                outer_steps=int(policy_config.get("outer_steps", 10)),
                inner_steps=1,
            ).eval(),
            {
                "method": method,
                "checkpoint": locator,
                "checkpoint_sha256": hashlib.sha256(locator.encode("utf-8")).hexdigest(),
                "checkpoint_identity_kind": "sha256-of-immutable-hub-locator",
                "model_revision": student.model_revision,
                "processor_revision": student.processor_revision,
            },
        )
    checkpoint = project_path(policy_config["checkpoint"])
    actual_sha = file_sha256(checkpoint)
    expected_sha = policy_config["checkpoint_sha256"]
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
    policy = InferencePolicy(
        student,
        method=method,
        outer_steps=int(policy_config["outer_steps"]),
        inner_steps=int(policy_config.get("inner_steps", 1)),
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
    }


__all__ = ["InferencePolicy", "load_inference_policy"]
