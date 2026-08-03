from __future__ import annotations

from typing import Any

import numpy as np
import torch

from tmd_policy.compatibility.actions import CanonicalActionSpace
from tmd_policy.data.schemas import TeacherQuery

from .cache import TeacherQueryCache


class FrozenTeacherQuerier:
    def __init__(
        self,
        policy: Any,
        preprocessor: Any,
        postprocessor: Any,
        cache: TeacherQueryCache,
        *,
        checkpoint: str,
        revision: str,
        processor_revision: str,
        inference_steps: int = 10,
    ) -> None:
        self.policy = policy.eval().requires_grad_(False)
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.cache = cache
        self.checkpoint = checkpoint
        self.revision = revision
        self.processor_revision = processor_revision
        self.inference_steps = inference_steps
        self._apply_inference_steps()

    def _apply_inference_steps(self) -> None:
        configs: list[Any] = []
        if hasattr(self.policy, "config"):
            configs.append(self.policy.config)
        model = getattr(self.policy, "model", None)
        if model is not None and hasattr(model, "config") and model.config not in configs:
            configs.append(model.config)
        if not configs:
            raise RuntimeError("teacher exposes no runtime config.num_inference_steps")
        for config in configs:
            if not hasattr(config, "num_inference_steps"):
                raise RuntimeError("teacher runtime config lacks num_inference_steps")
            config.num_inference_steps = self.inference_steps
            if int(config.num_inference_steps) != self.inference_steps:
                raise RuntimeError("teacher rejected configured num_inference_steps")

    def query(
        self,
        canonical_observation: dict[str, Any],
        *,
        observation_id: str,
        task_index: int,
        instruction: str,
        sampling_seed: int,
        sample_index: int = 0,
    ) -> np.ndarray:
        cached = self.cache.get(
            observation_id,
            self.checkpoint,
            self.revision,
            self.processor_revision,
            sampling_seed,
            self.inference_steps,
            sample_index,
        )
        if cached is not None:
            return np.asarray(cached[1]["action_chunk"], dtype=np.float32)
        batch = dict(canonical_observation)
        batch["task"] = [instruction]
        processed = self.preprocessor(batch)
        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.inference_mode(), torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(sampling_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(sampling_seed)
            actions = self.policy.predict_action_chunk(processed)
            canonical = self.postprocessor(actions)
        canonical_tensor = CanonicalActionSpace().project(torch.as_tensor(canonical).detach().cpu())
        canonical_np = canonical_tensor.numpy()[0].astype(np.float32)
        cache_key = TeacherQuery.make_cache_key(
            observation_id,
            self.checkpoint,
            self.revision,
            self.processor_revision,
            self.inference_steps,
            sampling_seed,
            sample_index,
        )
        query = TeacherQuery(
            sample_id=f"teacher-{cache_key[:24]}",
            observation_id=observation_id,
            task_index=task_index,
            instruction=instruction,
            teacher_checkpoint=self.checkpoint,
            teacher_revision=self.revision,
            processor_revision=self.processor_revision,
            sampling_seed=sampling_seed,
            inference_steps=self.inference_steps,
            sample_index=sample_index,
            action_chunk=canonical_np,
            action_valid=np.ones(canonical_np.shape[0], dtype=bool),
        )
        self.cache.put(query)
        return canonical_np
