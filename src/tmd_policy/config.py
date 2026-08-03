from __future__ import annotations

import json
import re
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from types import UnionType
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from tmd_policy.models.transition_head import InnerSourceMode


def _meta(
    valid: str,
    meaning: str,
    consumer: str,
    *,
    cache: bool = False,
    checkpoint: bool = False,
    affects: str = "training",
    recommended: str,
) -> dict[str, Any]:
    return {
        "valid": valid,
        "meaning": meaning,
        "consumer": consumer,
        "cache": cache,
        "checkpoint": checkpoint,
        "affects": affects,
        "recommended": recommended,
    }


@dataclass(frozen=True)
class HorizonConfig:
    prediction_horizon: int = field(
        default=50,
        metadata=_meta(
            "exactly 50 for LIBERO",
            "Number of actions in each planned chunk H_plan.",
            "data.schemas.ExpertChunk; rollout.collector.collect_rollout_episode",
            cache=True,
            checkpoint=True,
            affects="training,inference,storage",
            recommended="50",
        ),
    )
    execution_horizon: int = field(
        default=10,
        metadata=_meta(
            "1..prediction_horizon; exactly 10 for LIBERO",
            "Maximum real environment transitions before replanning H_exec.",
            "data.schemas.RolloutChunk; rollout.collector.collect_rollout_episode",
            cache=True,
            checkpoint=True,
            affects="training,evaluation,storage",
            recommended="10",
        ),
    )

    def __post_init__(self) -> None:
        if self.prediction_horizon != 50 or self.execution_horizon != 10:
            raise ValueError("the audited LIBERO contract requires horizons prediction=50, execution=10")


@dataclass(frozen=True)
class CanonicalConfig:
    state_dim: int = field(
        default=8,
        metadata=_meta(
            "exactly 8",
            "Canonical LIBERO low-dimensional state width.",
            "compatibility.actions.StateCompatibilityAdapter; data.schemas",
            cache=True,
            checkpoint=True,
            affects="training,inference,storage",
            recommended="8",
        ),
    )
    action_dim: int = field(
        default=7,
        metadata=_meta(
            "exactly 7",
            "Canonical LIBERO action width.",
            "compatibility.actions.CanonicalActionSpace; data.schemas",
            cache=True,
            checkpoint=True,
            affects="training,inference,storage",
            recommended="7",
        ),
    )
    action_min: float = field(
        default=-1.0,
        metadata=_meta(
            "exactly -1.0",
            "Inclusive lower bound after official postprocessing.",
            "compatibility.actions.CanonicalActionSpace; data.schemas._canonical_actions",
            cache=True,
            affects="inference,storage",
            recommended="-1.0",
        ),
    )
    action_max: float = field(
        default=1.0,
        metadata=_meta(
            "exactly 1.0",
            "Inclusive upper bound after official postprocessing.",
            "compatibility.actions.CanonicalActionSpace; data.schemas._canonical_actions",
            cache=True,
            affects="inference,storage",
            recommended="1.0",
        ),
    )

    def __post_init__(self) -> None:
        if (self.state_dim, self.action_dim) != (8, 7):
            raise ValueError("canonical LIBERO dimensions must be state=8 and action=7")
        if (self.action_min, self.action_max) != (-1.0, 1.0):
            raise ValueError("canonical LIBERO action bounds must be [-1, 1]")


@dataclass(frozen=True)
class CheckpointConfig:
    student_id: str = field(
        default="lerobot/smolvla_libero",
        metadata=_meta(
            "nonempty Hub ID or local path",
            "Frozen SmolVLA base checkpoint.",
            "models.smolvla_tmd.load_smolvla_tmd",
            cache=True,
            checkpoint=True,
            affects="training,inference,evaluation",
            recommended="lerobot/smolvla_libero",
        ),
    )
    student_revision: str = field(
        default="31d453f7edd78c839a8bbc39744a292686daf0de",
        metadata=_meta(
            "40-character git commit",
            "Immutable student model revision.",
            "models.smolvla_tmd.load_smolvla_tmd",
            cache=True,
            checkpoint=True,
            affects="training,inference,evaluation",
            recommended="31d453f7edd78c839a8bbc39744a292686daf0de",
        ),
    )
    student_processor_revision: str = field(
        default="31d453f7edd78c839a8bbc39744a292686daf0de",
        metadata=_meta(
            "40-character git commit",
            "Immutable student pre/postprocessor revision.",
            "models.smolvla_tmd.load_smolvla_tmd",
            cache=True,
            checkpoint=True,
            affects="training,inference,evaluation",
            recommended="same as student_revision",
        ),
    )
    teacher_id: str = field(
        default="lerobot/pi05_libero_finetuned",
        metadata=_meta(
            "nonempty Hub ID or local path",
            "Frozen pi0.5 teacher checkpoint; unused in B0-B2.",
            "teacher.query.FrozenTeacherQuerier",
            cache=True,
            checkpoint=True,
            affects="training",
            recommended="lerobot/pi05_libero_finetuned",
        ),
    )
    teacher_revision: str = field(
        default="8e174154ef5f6c60a8da12ae99c303d8963138c1",
        metadata=_meta(
            "40-character git commit",
            "Immutable teacher model revision.",
            "teacher.query.FrozenTeacherQuerier",
            cache=True,
            checkpoint=True,
            affects="training",
            recommended="8e174154ef5f6c60a8da12ae99c303d8963138c1",
        ),
    )
    teacher_processor_revision: str = field(
        default="8e174154ef5f6c60a8da12ae99c303d8963138c1",
        metadata=_meta(
            "40-character git commit",
            "Immutable teacher pre/postprocessor revision included in cache identity.",
            "teacher.query.FrozenTeacherQuerier; teacher.cache.TeacherQueryCache",
            cache=True,
            checkpoint=True,
            affects="training,storage",
            recommended="same as teacher_revision",
        ),
    )
    teacher_inference_steps: int = field(
        default=10,
        metadata=_meta(
            "positive integer",
            "Actual number of teacher denoising steps.",
            "teacher.query.FrozenTeacherQuerier._apply_inference_steps",
            cache=True,
            checkpoint=True,
            affects="training",
            recommended="10",
        ),
    )
    lerobot_commit: str = field(
        default="3b2c0e59d557be5bd60ae4b1a6b51ba44936ebf6",
        metadata=_meta(
            "40-character git commit",
            "Required source revision for the local LeRobot dependency.",
            "compatibility.lerobot_api.verify_lerobot_api",
            checkpoint=True,
            affects="training,inference,evaluation",
            recommended="3b2c0e59d557be5bd60ae4b1a6b51ba44936ebf6",
        ),
    )

    def __post_init__(self) -> None:
        for name in ("student_id", "teacher_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be nonempty")
        for name in (
            "student_revision",
            "student_processor_revision",
            "teacher_revision",
            "teacher_processor_revision",
            "lerobot_commit",
        ):
            if re.fullmatch(r"[0-9a-f]{40}", getattr(self, name)) is None:
                raise ValueError(f"{name} must be a pinned 40-character lowercase git commit")
        if self.teacher_inference_steps < 1:
            raise ValueError("teacher_inference_steps must be positive")


@dataclass(frozen=True)
class DatasetConfig:
    task_suite: str = field(
        default="libero_10",
        metadata=_meta(
            "libero_spatial, libero_object, libero_goal, libero_10, or libero_90",
            "LIBERO benchmark suite used for complete-episode evaluation.",
            "evaluation.policy_runner.evaluate_policy_arm",
            cache=True,
            affects="evaluation,storage",
            recommended="libero_10",
        ),
    )
    repo_id: str = field(
        default="HuggingFaceVLA/libero",
        metadata=_meta(
            "nonempty Hub dataset ID",
            "Expert dataset identity.",
            "data.expert.load_lerobot_expert_dataset",
            cache=True,
            checkpoint=True,
            affects="training,evaluation,storage",
            recommended="HuggingFaceVLA/libero",
        ),
    )
    revision: str = field(
        default="86958911c0f959db2bbbdb107eb3e17c5f9c798e",
        metadata=_meta(
            "40-character git commit",
            "Immutable expert dataset revision.",
            "data.expert.load_lerobot_expert_dataset",
            cache=True,
            checkpoint=True,
            affects="training,evaluation,storage",
            recommended="86958911c0f959db2bbbdb107eb3e17c5f9c798e",
        ),
    )
    split_seed: int = field(
        default=17,
        metadata=_meta(
            "nonnegative integer",
            "Seed for task-stratified episode splitting.",
            "data.expert.episode_split_three_way",
            cache=True,
            affects="training,evaluation,storage",
            recommended="17",
        ),
    )
    validation_fraction: float = field(
        default=0.1,
        metadata=_meta(
            "0 <= value < 1",
            "Per-task fraction of complete episodes assigned to validation.",
            "data.expert.episode_split_three_way",
            cache=True,
            affects="training,evaluation,storage",
            recommended="0.1",
        ),
    )
    test_fraction: float = field(
        default=0.1,
        metadata=_meta(
            "0 <= value < 1 and validation+test < 1",
            "Per-task fraction of complete episodes assigned to test.",
            "data.expert.episode_split_three_way",
            cache=True,
            affects="training,evaluation,storage",
            recommended="0.1",
        ),
    )
    stride: int = field(
        default=10,
        metadata=_meta(
            "positive integer",
            "Frame stride between expert chunk starts.",
            "data.expert.build_expert_chunks",
            cache=True,
            affects="training,storage",
            recommended="10",
        ),
    )
    episodes: tuple[int, ...] = field(
        default=(0, 1),
        metadata=_meta(
            "nonempty unique nonnegative integers",
            "Complete expert episodes placed in scope.",
            "data.expert.load_lerobot_expert_dataset",
            cache=True,
            affects="training,evaluation,storage",
            recommended="[0, 1] for smoke; expand for experiments",
        ),
    )

    def __post_init__(self) -> None:
        if self.task_suite not in {
            "libero_spatial",
            "libero_object",
            "libero_goal",
            "libero_10",
            "libero_90",
        }:
            raise ValueError("unsupported LIBERO task suite")
        if not self.repo_id.strip() or re.fullmatch(r"[0-9a-f]{40}", self.revision) is None:
            raise ValueError("dataset repo_id must be nonempty and revision must be a 40-character commit")
        if (
            self.split_seed < 0
            or not 0 <= self.validation_fraction < 1
            or not 0 <= self.test_fraction < 1
            or self.validation_fraction + self.test_fraction >= 1
            or self.stride < 1
        ):
            raise ValueError("invalid dataset split seed, fraction, or stride")
        if not self.episodes or len(set(self.episodes)) != len(self.episodes) or min(self.episodes) < 0:
            raise ValueError("episodes must be nonempty, unique, and nonnegative")


@dataclass(frozen=True)
class TMDConfig:
    outer_steps: int = field(
        default=2,
        metadata=_meta(
            "positive integer",
            "Number of outer Euler evaluations.",
            "models.tmd.TMDActionGenerator",
            checkpoint=True,
            affects="training,inference,evaluation",
            recommended="2",
        ),
    )
    inner_steps: int = field(
        default=2,
        metadata=_meta(
            "positive integer",
            "Transition-head Euler evaluations per outer step.",
            "models.tmd.TMDActionGenerator",
            checkpoint=True,
            affects="training,inference,evaluation",
            recommended="2",
        ),
    )
    inner_source_mode: str = field(
        default=InnerSourceMode.GAUSSIAN_TM.value,
        metadata=_meta(
            "gaussian_tm or anchored_tm_ablation; meanflow reserved",
            "Distribution and parameterization of the inner transition flow.",
            "models.transition_head.RecurrentTransitionHead",
            checkpoint=True,
            affects="training,inference,evaluation",
            recommended="gaussian_tm",
        ),
    )
    recurrent_layers: int = field(
        default=2,
        metadata=_meta(
            "positive integer",
            "Number of recurrent GRUCell layers in the transition head.",
            "models.transition_head.RecurrentTransitionHead",
            checkpoint=True,
            affects="training,inference",
            recommended="2",
        ),
    )
    hidden_dim: int = field(
        default=256,
        metadata=_meta(
            "even integer >= 2",
            "Transition-head recurrent width.",
            "models.transition_head.RecurrentTransitionHead",
            checkpoint=True,
            affects="training,inference",
            recommended="256",
        ),
    )
    dropout: float = field(
        default=0.0,
        metadata=_meta(
            "0 <= value < 1",
            "Transition-head inter-layer dropout probability.",
            "models.transition_head.RecurrentTransitionHead",
            checkpoint=True,
            affects="training,inference",
            recommended="0.0",
        ),
    )
    loss: str = field(
        default="huber",
        metadata=_meta(
            "huber or mse",
            "Elementwise transition-velocity regression loss.",
            "models.transition_head.RecurrentTransitionHead.matching_loss",
            checkpoint=True,
            affects="training",
            recommended="huber",
        ),
    )
    main_loss_weight: float = field(
        default=0.0,
        metadata=_meta(
            "nonnegative float",
            "Coefficient on optional direct backbone flow loss.",
            "models.tmd.TMDActionGenerator.transition_matching_loss",
            checkpoint=True,
            affects="training",
            recommended="0.0",
        ),
    )
    freeze_vision_encoder: bool = field(
        default=True,
        metadata=_meta(
            "true only in the audited first experiment",
            "Whether the large vision/language backbone remains frozen.",
            "models.smolvla_tmd.SmolVLATMDPolicy.configure_trainable",
            checkpoint=True,
            affects="training",
            recommended="true",
        ),
    )
    train_main_action_projections: bool = field(
        default=False,
        metadata=_meta(
            "boolean",
            "Whether small SmolVLA action/state/time projection modules are trainable.",
            "models.smolvla_tmd.SmolVLATMDPolicy.configure_trainable",
            checkpoint=True,
            affects="training",
            recommended="false for B2",
        ),
    )

    def __post_init__(self) -> None:
        if self.outer_steps < 1 or self.inner_steps < 1 or self.recurrent_layers < 1:
            raise ValueError("outer_steps, inner_steps, and recurrent_layers must be positive")
        if self.hidden_dim < 2 or self.hidden_dim % 2:
            raise ValueError("hidden_dim must be an even integer >= 2")
        if not 0 <= self.dropout < 1 or self.main_loss_weight < 0:
            raise ValueError("invalid transition dropout or main loss weight")
        if self.loss not in {"huber", "mse"}:
            raise ValueError("tmd.loss must be huber or mse")
        try:
            mode = InnerSourceMode(self.inner_source_mode)
        except ValueError as error:
            raise ValueError(f"unknown tmd.inner_source_mode: {self.inner_source_mode}") from error
        if mode is InnerSourceMode.GAUSSIAN_TM_MEANFLOW:
            raise ValueError("gaussian_tm_meanflow is reserved and unsupported")
        if not self.freeze_vision_encoder:
            raise ValueError("unfreezing the vision encoder is unsupported in the audited experiment")


@dataclass(frozen=True)
class DiscriminatorConfig:
    model_dim: int = field(
        default=128,
        metadata=_meta(
            "positive and divisible by num_heads",
            "Causal path-transformer embedding width.",
            "models.discriminator.CausalPathDiscriminator",
            checkpoint=True,
            affects="training,evaluation",
            recommended="128",
        ),
    )
    num_layers: int = field(
        default=3,
        metadata=_meta(
            "positive integer",
            "Transformer encoder depth.",
            "models.discriminator.CausalPathDiscriminator",
            checkpoint=True,
            affects="training,evaluation",
            recommended="3",
        ),
    )
    num_heads: int = field(
        default=4,
        metadata=_meta(
            "positive divisor of model_dim",
            "Self-attention head count.",
            "models.discriminator.CausalPathDiscriminator",
            checkpoint=True,
            affects="training,evaluation",
            recommended="4",
        ),
    )
    feedforward_dim: int = field(
        default=256,
        metadata=_meta(
            "positive integer",
            "Transformer feed-forward width.",
            "models.discriminator.CausalPathDiscriminator",
            checkpoint=True,
            affects="training,evaluation",
            recommended="256",
        ),
    )
    dropout: float = field(
        default=0.1,
        metadata=_meta(
            "0 <= value < 1",
            "Discriminator dropout probability.",
            "models.discriminator.CausalPathDiscriminator",
            checkpoint=True,
            affects="training,evaluation",
            recommended="0.1",
        ),
    )
    num_tasks: int = field(
        default=40,
        metadata=_meta(
            "positive integer",
            "Size of the task embedding table.",
            "models.discriminator.CausalPathDiscriminator",
            checkpoint=True,
            affects="training,evaluation",
            recommended="40",
        ),
    )
    weighting_min: float = field(
        default=0.5,
        metadata=_meta(
            "positive and <= weighting_max",
            "Minimum detached mismatch-emphasis weight.",
            "training.distillation.MismatchPrioritization",
            checkpoint=True,
            affects="training",
            recommended="0.5",
        ),
    )
    weighting_max: float = field(
        default=2.0,
        metadata=_meta(
            ">= weighting_min",
            "Maximum detached mismatch-emphasis weight.",
            "training.distillation.MismatchPrioritization",
            checkpoint=True,
            affects="training",
            recommended="2.0",
        ),
    )
    weighting_temperature: float = field(
        default=1.0,
        metadata=_meta(
            "positive float",
            "Temperature of mismatch-emphasis sigmoid.",
            "training.distillation.MismatchPrioritization",
            checkpoint=True,
            affects="training",
            recommended="1.0",
        ),
    )

    def __post_init__(self) -> None:
        if min(self.model_dim, self.num_layers, self.num_heads, self.feedforward_dim, self.num_tasks) < 1:
            raise ValueError("discriminator dimensions and counts must be positive")
        if self.model_dim % self.num_heads:
            raise ValueError("discriminator.model_dim must be divisible by num_heads")
        if not 0 <= self.dropout < 1:
            raise ValueError("discriminator.dropout must be in [0,1)")
        if self.weighting_min <= 0 or self.weighting_max < self.weighting_min:
            raise ValueError("invalid discriminator weighting bounds")
        if self.weighting_temperature <= 0:
            raise ValueError("weighting_temperature must be positive")


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = field(
        default=7,
        metadata=_meta(
            "nonnegative integer",
            "Primary training RNG seed.",
            "training.runner.seed_everything",
            checkpoint=True,
            affects="training",
            recommended="7",
        ),
    )
    evaluation_seeds: tuple[int, ...] = field(
        default=(7, 17, 27),
        metadata=_meta(
            "nonempty unique nonnegative integers",
            "Complete-episode evaluation reset seeds.",
            "evaluation.policy_runner.evaluate_policy_arm",
            affects="evaluation",
            recommended="[7, 17, 27]",
        ),
    )
    task_indices: tuple[int, ...] = field(
        default=(0,),
        metadata=_meta(
            "nonempty unique nonnegative integers",
            "LIBERO task subset for the run.",
            "evaluation.policy_runner.evaluate_policy_arm",
            cache=True,
            affects="training,evaluation,storage",
            recommended="[0] for the first experiment",
        ),
    )
    batch_size: int = field(
        default=8,
        metadata=_meta(
            "positive integer",
            "Training minibatch size.",
            "training.runner.train_expert_chunk",
            affects="training",
            recommended="8",
        ),
    )
    expert_steps: int = field(
        default=20,
        metadata=_meta(
            "nonnegative integer",
            "Expert-only transition-head update count.",
            "training.runner.train_expert_chunk",
            affects="training",
            recommended="20 smoke; increase for B2",
        ),
    )
    discriminator_steps: int = field(
        default=50,
        metadata=_meta(
            "nonnegative integer",
            "Discriminator optimizer update count.",
            "cli._train_discriminator; experiments.motivation.runner.run_experiments",
            affects="training",
            recommended="50 smoke",
        ),
    )
    distillation_steps: int = field(
        default=20,
        metadata=_meta(
            "nonnegative integer; gated until B0-B2",
            "Teacher-distillation update count.",
            "cli._gated",
            affects="training",
            recommended="0 until B0-B2 pass",
        ),
    )
    learning_rate: float = field(
        default=1e-5,
        metadata=_meta(
            "positive float",
            "AdamW learning rate.",
            "training.runner.make_optimizer",
            checkpoint=True,
            affects="training",
            recommended="1e-5 for the first real-chunk experiment",
        ),
    )
    weight_decay: float = field(
        default=1e-4,
        metadata=_meta(
            "nonnegative float",
            "AdamW decoupled weight decay.",
            "training.runner.make_optimizer",
            checkpoint=True,
            affects="training",
            recommended="1e-4",
        ),
    )
    expert_weight: float = field(
        default=1.0,
        metadata=_meta(
            "nonnegative float",
            "Coefficient on expert per-sample loss.",
            "cli._gated; training.distillation.combined_distillation_loss",
            checkpoint=True,
            affects="training",
            recommended="1.0",
        ),
    )
    teacher_weight: float = field(
        default=1.0,
        metadata=_meta(
            "nonnegative float",
            "Coefficient on teacher per-sample loss; gated until B3.",
            "cli._gated; training.distillation.combined_distillation_loss",
            checkpoint=True,
            affects="training",
            recommended="1.0 when B3 is enabled",
        ),
    )
    replay_mode: str = field(
        default="fresh_only",
        metadata=_meta(
            "fresh_only or historical_importance",
            "Whether discriminator negatives are exact current samples or corrected replay.",
            "cli._gated; training.replay.ReplayPools",
            checkpoint=True,
            affects="training,storage",
            recommended="fresh_only",
        ),
    )
    minimum_fresh_current: int = field(
        default=1,
        metadata=_meta(
            "positive integer",
            "Minimum exact-current records before discriminator training.",
            "cli._gated; training.replay.ReplayPools.require_fresh_current",
            affects="training,storage",
            recommended="1 smoke; increase for experiments",
        ),
    )
    device: str = field(
        default="cuda",
        metadata=_meta(
            "cpu, cuda, or cuda:N",
            "Torch execution device.",
            "models.smolvla_tmd.load_smolvla_tmd; training.runner.train_expert_chunk",
            affects="training,inference,evaluation",
            recommended="cuda",
        ),
    )

    def __post_init__(self) -> None:
        if self.seed < 0 or self.batch_size < 1:
            raise ValueError("training seed must be nonnegative and batch_size positive")
        for name in ("evaluation_seeds", "task_indices"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values) or min(values) < 0:
                raise ValueError(f"{name} must be nonempty, unique, and nonnegative")
        if min(self.expert_steps, self.discriminator_steps, self.distillation_steps) < 0:
            raise ValueError("training step counts must be nonnegative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay nonnegative")
        if self.expert_weight < 0 or self.teacher_weight < 0:
            raise ValueError("training loss coefficients must be nonnegative")
        if self.replay_mode not in {"fresh_only", "historical_importance"}:
            raise ValueError("unsupported replay_mode")
        if self.minimum_fresh_current < 1:
            raise ValueError("minimum_fresh_current must be positive")
        if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", self.device):
            raise ValueError("device must be cpu, cuda, or cuda:N")


@dataclass(frozen=True)
class EvaluationConfig:
    episode_length: int = field(
        default=20,
        metadata=_meta(
            "positive integer",
            "Explicit environment time limit for each complete first-experiment episode.",
            "evaluation.policy_runner.evaluate_policy_arm",
            cache=True,
            affects="evaluation,storage",
            recommended="20 for plumbing feasibility only; use suite default for policy claims",
        ),
    )
    bootstrap_resamples: int = field(
        default=1000,
        metadata=_meta(
            "positive integer",
            "Number of task-stratified complete-episode bootstrap resamples.",
            "evaluation.metrics.bootstrap_episode_statistic",
            affects="evaluation",
            recommended="1000",
        ),
    )
    bootstrap_confidence: float = field(
        default=0.95,
        metadata=_meta(
            "0 < value < 1",
            "Percentile-bootstrap confidence level.",
            "evaluation.metrics.bootstrap_episode_statistic",
            affects="evaluation",
            recommended="0.95",
        ),
    )

    def __post_init__(self) -> None:
        if self.episode_length < 1 or self.bootstrap_resamples < 1:
            raise ValueError("evaluation episode length and bootstrap resamples must be positive")
        if not 0 < self.bootstrap_confidence < 1:
            raise ValueError("bootstrap_confidence must be in (0,1)")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = field(
        default="tiny_libero_tmd",
        metadata=_meta(
            "nonempty filesystem-safe name",
            "Run identity used for artifact directories.",
            "cli._audit; config.save_resolved_config",
            affects="storage",
            recommended="tiny_libero_tmd",
        ),
    )
    horizons: HorizonConfig = field(default_factory=HorizonConfig)
    canonical: CanonicalConfig = field(default_factory=CanonicalConfig)
    checkpoints: CheckpointConfig = field(default_factory=CheckpointConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tmd: TMDConfig = field(default_factory=TMDConfig)
    discriminator: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def __post_init__(self) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name) is None:
            raise ValueError("experiment name must be nonempty and filesystem-safe")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _construct[T](cls: type[T], values: dict[str, Any], path: str = "") -> T:
    if not isinstance(values, dict):
        raise TypeError(f"{path or 'configuration root'} must be a mapping")
    known = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        location = path or "configuration root"
        raise KeyError(f"unknown configuration key(s) at {location}: {', '.join(unknown)}")
    kwargs: dict[str, Any] = {}
    hints = get_type_hints(cls)
    for item in fields(cls):
        if item.name not in values:
            continue
        value = values[item.name]
        item_type = hints[item.name]
        if is_dataclass(item_type):
            value = _construct(item_type, value, f"{path}.{item.name}".strip("."))
        elif get_origin(item_type) is tuple and isinstance(value, list):
            value = tuple(value)
        kwargs[item.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    if not isinstance(raw, dict):
        raise TypeError("configuration root must be a mapping")
    return _construct(ExperimentConfig, raw)


def save_resolved_config(config: ExperimentConfig, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    return target


def _type_name(annotation: Any) -> str:
    origin = get_origin(annotation)
    if origin is tuple:
        return "tuple[" + ", ".join(_type_name(arg) for arg in get_args(annotation)) + "]"
    if origin in (UnionType,):
        return " | ".join(_type_name(arg) for arg in get_args(annotation))
    return getattr(annotation, "__name__", str(annotation).replace("typing.", ""))


def config_field_records() -> list[dict[str, Any]]:
    """Return one generated documentation/consumption record per leaf field."""

    records: list[dict[str, Any]] = []

    def visit(cls: type[Any], prefix: str = "") -> None:
        hints = get_type_hints(cls)
        for item in fields(cls):
            dotted = f"{prefix}.{item.name}".strip(".")
            annotation = hints[item.name]
            if is_dataclass(annotation):
                visit(annotation, dotted)
                continue
            if not item.metadata or not item.metadata.get("consumer"):
                raise RuntimeError(f"config field {dotted} has no executable consumer metadata")
            if item.default is not MISSING:
                default = item.default
            elif item.default_factory is not MISSING:
                default = item.default_factory()
            else:
                default = None
            records.append(
                {
                    "name": dotted,
                    "type": _type_name(annotation),
                    "default": default,
                    **dict(item.metadata),
                }
            )

    visit(ExperimentConfig)
    return records


def config_runtime_report() -> dict[str, str]:
    """Map every supported field to the executable consumer that owns it."""

    return {record["name"]: record["consumer"] for record in config_field_records()}


def render_config_reference() -> str:
    rows = [
        "# Generated configuration reference",
        "",
        "This file is generated from `tmd_policy.config`; edit the dataclasses, not this file.",
        "",
        "| Field | Type | Default | Valid values | Mathematical meaning | Consumer | Cache? | Checkpoint? | Affects | Recommended |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in config_field_records():
        default = json.dumps(record["default"])
        values = [
            record["name"],
            record["type"],
            default,
            record["valid"],
            record["meaning"],
            record["consumer"],
            "yes" if record["cache"] else "no",
            "yes" if record["checkpoint"] else "no",
            record["affects"],
            record["recommended"],
        ]
        rows.append("| " + " | ".join(str(value).replace("|", "\\|") for value in values) + " |")
    return "\n".join(rows) + "\n"


__all__ = [
    "CanonicalConfig",
    "CheckpointConfig",
    "DatasetConfig",
    "DiscriminatorConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "HorizonConfig",
    "TMDConfig",
    "TrainingConfig",
    "config_field_records",
    "config_runtime_report",
    "load_config",
    "render_config_reference",
    "save_resolved_config",
]
