from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from torch import Tensor, nn

from tmd_policy.config import ExperimentConfig, save_resolved_config
from tmd_policy.data.storage import ChunkStore
from tmd_policy.evaluation.metrics import bootstrap_episode_statistic
from tmd_policy.models.smolvla_tmd import load_smolvla_tmd
from tmd_policy.rollout.collector import CanonicalChunkRunner, collect_rollout_episode
from tmd_policy.training.checkpoint import load_policy_for_inference


class OfficialSmolVLAAdapter(nn.Module):
    """Expose the same seeded chunk interface for the official 10-step B0 arm."""

    uses_inner_noise = False

    def __init__(self, base_policy: nn.Module) -> None:
        super().__init__()
        self.base_policy = base_policy
        self.config = base_policy.config
        self.generator = SimpleNamespace(outer_steps=10)

    def reset(self) -> None:
        self.base_policy.reset()

    def predict_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        *,
        inner_noises: Tensor | None = None,
    ) -> Tensor:
        if inner_noises is not None:
            raise ValueError("the official B0 arm has no inner transition flow")
        return self.base_policy.predict_action_chunk(batch, noise=noise)


def _load_arm(
    config: ExperimentConfig,
    arm: str,
    checkpoint: str | Path | None,
) -> tuple[nn.Module, Any, Any, dict[str, Any]]:
    policy, preprocessor, postprocessor = load_smolvla_tmd(
        config.checkpoints.student_id,
        revision=config.checkpoints.student_revision,
        processor_revision=config.checkpoints.student_processor_revision,
        lerobot_commit=config.checkpoints.lerobot_commit,
        device=config.training.device,
        outer_steps=config.tmd.outer_steps,
        inner_steps=config.tmd.inner_steps,
        recurrent_layers=config.tmd.recurrent_layers,
        hidden_dim=config.tmd.hidden_dim,
        main_loss_weight=config.tmd.main_loss_weight,
        inner_source_mode=config.tmd.inner_source_mode,
    )
    metadata: dict[str, Any] = {
        "base_checkpoint": config.checkpoints.student_id,
        "base_revision": config.checkpoints.student_revision,
    }
    if arm == "B0":
        policy.base_policy.model.config.num_steps = 10
        return OfficialSmolVLAAdapter(policy.base_policy).eval(), preprocessor, postprocessor, metadata
    if arm == "B1":
        if checkpoint is not None:
            raise ValueError("B1 is the zero-initialized, untrained head and takes no checkpoint")
        return policy.eval(), preprocessor, postprocessor, metadata
    if arm == "B2":
        if checkpoint is None:
            raise ValueError("B2 evaluation requires --checkpoint from train-expert")
        metadata = load_policy_for_inference(checkpoint, policy)
        expected = {
            "outer_steps": config.tmd.outer_steps,
            "inner_steps": config.tmd.inner_steps,
            "inner_source_mode": config.tmd.inner_source_mode,
            "base_revision": config.checkpoints.student_revision,
            "dataset_revision": config.dataset.revision,
        }
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"B2 checkpoint/config mismatch: {mismatches}")
        return policy.eval(), preprocessor, postprocessor, metadata
    raise ValueError("arm must be B0, B1, or B2; B3/B4 are gated")


def evaluate_policy_arm(
    config: ExperimentConfig,
    *,
    arm: str,
    output_dir: str | Path,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Run complete episodes for one permitted first-experiment policy arm."""

    arm = arm.upper()
    if arm not in {"B0", "B1", "B2"}:
        raise RuntimeError("only B0, B1, and B2 are executable before the teacher gate")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    save_resolved_config(config, output / "resolved_config.yaml")
    policy, preprocessor, postprocessor, checkpoint_metadata = _load_arm(
        config, arm, checkpoint
    )
    from lerobot.envs.configs import LiberoEnv

    env_config = LiberoEnv(
        task=config.dataset.task_suite,
        task_ids=list(config.training.task_indices),
        observation_height=256,
        observation_width=256,
        auto_reset_on_done=False,
        episode_length=config.evaluation.episode_length,
    )
    env_preprocessor, _ = env_config.get_env_processors()
    environments = env_config.create_envs(n_envs=1, use_async_envs=False)
    runner = CanonicalChunkRunner(policy, preprocessor, postprocessor)
    store = ChunkStore(output / "rollout_chunks")
    episodes: list[dict[str, Any]] = []
    try:
        for task_index in config.training.task_indices:
            env = environments[config.dataset.task_suite][task_index]
            instruction_values = env.call("task_description")
            instruction = str(instruction_values[0])
            for reset_seed in config.training.evaluation_seeds:
                policy.reset()
                records, metrics = collect_rollout_episode(
                    env,
                    env_preprocessor,
                    runner,
                    policy_checkpoint=(
                        str(Path(checkpoint).resolve()) if checkpoint is not None else config.checkpoints.student_id
                    ),
                    policy_version=f"{arm}-task-{task_index}",
                    collection_round=0,
                    task_index=task_index,
                    instruction=instruction,
                    reset_seed=reset_seed,
                    base_noise_seed=100_000 * task_index + 1_000 * reset_seed + 17,
                    prediction_horizon=config.horizons.prediction_horizon,
                    execution_horizon=config.horizons.execution_horizon,
                    max_chunks=None,
                    max_environment_steps=config.evaluation.episode_length,
                )
                for record in records:
                    store.append(record)
                episodes.append(
                    {
                        "task_index": task_index,
                        "instruction": instruction,
                        "reset_seed": reset_seed,
                        **metrics,
                    }
                )
    finally:
        for suite in environments.values():
            for env in suite.values():
                env.close()
    successes = np.asarray([episode["success"] for episode in episodes], dtype=np.float64)
    task_ids = np.asarray([episode["task_index"] for episode in episodes], dtype=np.int64)
    latency = np.asarray([episode["total_model_latency_s"] for episode in episodes])
    report = {
        "arm": arm,
        "data_label": "real LIBERO complete-episode evaluation",
        "suite": config.dataset.task_suite,
        "task_indices": list(config.training.task_indices),
        "evaluation_seeds": list(config.training.evaluation_seeds),
        "episode_length": config.evaluation.episode_length,
        "episodes": episodes,
        "episode_count": len(episodes),
        "success_rate": float(successes.mean()),
        "success_bootstrap": bootstrap_episode_statistic(
            successes,
            task_ids=task_ids,
            confidence=config.evaluation.bootstrap_confidence,
            resamples=config.evaluation.bootstrap_resamples,
            seed=config.training.seed,
        ),
        "mean_total_model_latency_s_per_episode": float(latency.mean()),
        "checkpoint_metadata": checkpoint_metadata,
        "rollout_store": str((output / "rollout_chunks").resolve()),
    }
    (output / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


__all__ = ["OfficialSmolVLAAdapter", "evaluate_policy_arm"]
