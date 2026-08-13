"""Configuration loading for executable TMDpolicy runs.

YAML files are intentionally plain and fully resolved: there is no capability
registry and no mode that pretends to train without constructing the assets.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import yaml

LEROBOT_VERSION = "0.6.1"
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ConfigError(ValueError):
    """Raised before any model or dataset is loaded when a run is ambiguous."""


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required configuration field: {context}.{key}")
    return mapping[key]


def _revision(value: Any, field: str) -> str:
    value = str(value)
    if not IMMUTABLE_REVISION.fullmatch(value):
        raise ConfigError(f"{field} must be an immutable 40-character Hub commit, got {value!r}")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"unknown or deprecated configuration fields in {context}: {unknown}")


def _all_libero_tasks(benchmark: Any, context: str) -> None:
    if not isinstance(benchmark, list):
        raise ConfigError(f"{context} must be a suite/task list")
    expected = {
        "libero_spatial": set(range(10)),
        "libero_object": set(range(10)),
        "libero_goal": set(range(10)),
        "libero_10": set(range(10)),
    }
    actual: dict[str, set[int]] = {}
    for value in benchmark:
        suite = str(value.get("suite"))
        if suite in actual:
            raise ConfigError(f"{context} repeats suite {suite}")
        actual[suite] = {int(task) for task in value.get("task_ids", [])}
    if actual != expected:
        raise ConfigError(f"{context} must cover each of the four suites and all 40 tasks exactly")


def validate_config(config: dict[str, Any], *, expected_method: str | None = None) -> None:
    """Fail closed on asset identity and the shared LIBERO tensor contract."""

    method = str(_require(config, "method", "config"))
    root_fields = {
        "data_build_expert": {"method", "backend", "models", "dataset", "horizons", "output"},
        "pi05_flow_parity": {
            "method", "classification", "backend", "models", "dataset", "horizons", "parity", "output"
        },
        "flow_sft": {
            "method", "classification", "backend", "models", "dataset", "horizons", "training",
            "fine_tuning", "output"
        },
        "tmd_stage1": {
            "method", "classification", "backend", "models", "dataset", "horizons", "training",
            "tmd", "preflight", "output"
        },
        "dmd2_flow": {
            "method", "classification", "backend", "models", "dataset", "horizons", "training",
            "dmd2", "preflight", "output"
        },
        "tmd_stage2": {
            "method", "classification", "backend", "models", "dataset", "horizons", "training",
            "stage1_architecture", "stage2", "preflight", "output"
        },
        "occupancy_discriminator": {
            "method", "classification", "backend", "models", "dataset", "horizons", "training",
            "rollouts", "occupancy_discriminator", "preflight", "output"
        },
        "occupancy_tmd": {
            "method", "classification", "backend", "models", "dataset", "horizons", "training",
            "tmd", "occupancy", "output"
        },
        "collect_student": {
            "method", "classification", "backend", "models", "dataset", "horizons", "policy",
            "collection", "output"
        },
        "evaluate_libero": {
            "method", "classification", "backend", "models", "dataset", "horizons", "policy",
            "evaluation", "output"
        },
        "evaluate_libero_plus": {
            "method", "classification", "backend", "models", "dataset", "horizons", "policy",
            "evaluation", "output"
        },
    }
    if method not in root_fields:
        raise ConfigError(f"unknown production method: {method!r}")
    _reject_unknown(config, root_fields[method] | {"_config_path"}, "config")
    if expected_method is not None and method != expected_method:
        raise ConfigError(f"command expects method={expected_method!r}, config declares {method!r}")
    backend = _require(config, "backend", "config")
    _reject_unknown(backend, {"lerobot_version", "expected_source_hashes", "local_files_only"}, "backend")
    if _require(backend, "lerobot_version", "backend") != LEROBOT_VERSION:
        raise ConfigError(f"backend.lerobot_version must be exactly {LEROBOT_VERSION}")
    hashes = backend.get("expected_source_hashes")
    if hashes is not None and not isinstance(hashes, dict):
        raise ConfigError("backend.expected_source_hashes must be null or a module-name mapping")

    models = _require(config, "models", "config")
    _reject_unknown(models, {"student", "teacher"}, "models")
    for name in ("student", "teacher"):
        asset = _require(models, name, "models")
        _reject_unknown(asset, {"id", "revision", "processor_revision"}, f"models.{name}")
        if not str(_require(asset, "id", f"models.{name}")).strip():
            raise ConfigError(f"models.{name}.id must be nonempty")
        _revision(_require(asset, "revision", f"models.{name}"), f"models.{name}.revision")
        _revision(
            _require(asset, "processor_revision", f"models.{name}"),
            f"models.{name}.processor_revision",
        )

    dataset = _require(config, "dataset", "config")
    _reject_unknown(
        dataset,
        {
            "id", "revision", "cache", "manifest", "validation_fraction", "test_fraction",
            "split_seed", "download_videos", "video_backend",
        },
        "dataset",
    )
    if _require(dataset, "id", "dataset") != "lerobot/libero":
        raise ConfigError("the canonical expert dataset must be lerobot/libero")
    _revision(_require(dataset, "revision", "dataset"), "dataset.revision")
    fractions = [float(dataset.get("validation_fraction", 0.1)), float(dataset.get("test_fraction", 0.1))]
    if min(fractions) <= 0 or sum(fractions) >= 1:
        raise ConfigError("validation/test fractions must be positive and sum to less than one")

    horizons = _require(config, "horizons", "config")
    _reject_unknown(horizons, {"prediction", "execution"}, "horizons")
    if int(_require(horizons, "prediction", "horizons")) != 50:
        raise ConfigError("LIBERO prediction horizon is fixed at 50")
    execution = int(_require(horizons, "execution", "horizons"))
    if not 1 <= execution <= 50:
        raise ConfigError("horizons.execution must be in [1, 50]")

    training = config.get("training")
    if training is not None:
        _reject_unknown(
            training,
            {
                "seed", "device", "batch_size", "num_workers", "mixed_precision",
                "gradient_accumulation", "gradient_clip_norm", "learning_rate", "weight_decay",
                "betas", "epsilon", "warmup_steps", "minimum_lr_scale", "max_steps",
                "validation_interval", "validation_batches", "checkpoint_interval",
                "inference_checkpoint_interval", "diagnostics_interval",
            },
            "training",
        )
        if int(training.get("batch_size", 0)) < 1 or int(training.get("max_steps", 0)) < 1:
            raise ConfigError("training.batch_size and training.max_steps must be positive")
        if training.get("mixed_precision", "bf16") not in {"no", "fp16", "bf16"}:
            raise ConfigError("training.mixed_precision must be no, fp16, or bf16")
        if int(training.get("gradient_accumulation", 0)) < 1:
            raise ConfigError("training.gradient_accumulation must be positive")
        if int(training.get("inference_checkpoint_interval", 1)) < 1:
            raise ConfigError("training.inference_checkpoint_interval must be positive when set")
        if int(training.get("diagnostics_interval", 1)) < 1:
            raise ConfigError("training.diagnostics_interval must be positive")
    _reject_unknown(_require(config, "output", "config"), {"directory"}, "output")
    if "preflight" in config:
        _reject_unknown(config["preflight"], {"minimum_total_memory_gib"}, "preflight")

    if "capabilities" in json.dumps(config):
        raise ConfigError("capability declarations were removed; configure concrete assets instead")

    tmd_fields = {
        "variant", "attention_mode", "flow_matching_fraction", "normalization_constant_scale",
        "student_time_shift_gamma", "meanflow_time_shift_gamma", "condition_dropout_probability",
        "last_k_expert_blocks", "train_backbone_flow_blocks", "model_dim", "layers", "heads",
        "feedforward_dim", "dropout", "discrete_outer_steps", "discrete_inner_steps",
    }
    if method in {"tmd_stage1", "occupancy_tmd"}:
        tmd = _require(config, "tmd", "config")
        _reject_unknown(tmd, tmd_fields, "tmd")
        tmd_sections = [("tmd", tmd)]
    else:
        tmd_sections = []
    if method == "tmd_stage2":
        architecture = _require(config, "stage1_architecture", "config")
        _reject_unknown(architecture, tmd_fields, "stage1_architecture")
        tmd_sections.append(("stage1_architecture", architecture))
    for name, value in tmd_sections:
        if not 0 <= float(value["flow_matching_fraction"]) <= 1:
            raise ConfigError(f"{name}.flow_matching_fraction must be in [0,1]")
        if float(value.get("student_time_shift_gamma", 1.0)) < 1 or float(
            value.get("meanflow_time_shift_gamma", 1.0)
        ) < 1:
            raise ConfigError(f"{name} timestep shifts must be at least one")
        if float(value.get("condition_dropout_probability", 0.0)) != 0.0:
            raise ConfigError(
                f"{name}.condition_dropout_probability must be 0.0; this conditional baseline has no CFG"
            )
        if float(value["normalization_constant_scale"]) <= 0:
            raise ConfigError(f"{name}.normalization_constant_scale must be positive")
        if int(value["discrete_outer_steps"]) < 1 or int(value["discrete_inner_steps"]) < 1:
            raise ConfigError(f"{name} discrete student-grid step counts must be positive")

    if method in {"dmd2_flow", "tmd_stage2"}:
        section_name = "dmd2" if method == "dmd2_flow" else "stage2"
        objective = _require(config, section_name, "config")
        objective_fields = {
            "student_fine_tuning", "fake_score_variant", "fake_updates_per_generator",
            "student_training_mode",
            "generator_learning_rate", "fake_score_learning_rate",
            "discriminator_learning_rate", "gan_weight", "vsd_normalization_epsilon",
            "vsd_normalization",
            "teacher_device", "teacher_dtype", "fake_score_device",
            "vsd_time", "gan_time", "fake_score_time", "discriminator", "resource_estimate",
            "fake_score", "data_weight", "student_rollout_replay",
        }
        if method == "tmd_stage2":
            objective_fields |= {
                "stage1_checkpoint", "stage1_checkpoint_sha256",
                "discriminator_updates_per_generator",
            }
        else:
            objective_fields |= {
                "guidance_classifier_weight", "discriminator_updates_per_generator",
                "discrete_outer_steps", "student_time_shift_gamma",
            }
        _reject_unknown(objective, objective_fields, section_name)
        if "data_weight" in objective:
            raise ConfigError(
                f"{section_name}.data_weight is deprecated: faithful {method} has no SFT/data loss; "
                "migrate to the paper config or an explicitly named hybrid ablation"
            )
        expected_normalization = (
            "dmd2_teacher_residual_mean_abs"
            if method == "dmd2_flow"
            else "tmd_fake_teacher_difference_l1"
        )
        if objective.get("vsd_normalization") != expected_normalization:
            raise ConfigError(
                f"{section_name}.vsd_normalization must be {expected_normalization} for {method}"
            )
        expected_training_mode = (
            "backward_simulation_denoise_renoise"
            if method == "dmd2_flow"
            else "tmd_stage1_outer_transition"
        )
        if objective.get("student_training_mode") != expected_training_mode:
            raise ConfigError(
                f"{section_name}.student_training_mode must be {expected_training_mode}"
            )
        if int(objective.get("fake_updates_per_generator", 0)) != 5:
            raise ConfigError(f"{section_name}.fake_updates_per_generator must be 5")
        if method == "dmd2_flow":
            if int(objective["discrete_outer_steps"]) < 1:
                raise ConfigError("dmd2.discrete_outer_steps must be positive")
            if float(objective["student_time_shift_gamma"]) < 1:
                raise ConfigError("dmd2.student_time_shift_gamma must be at least one")
            replay = _require(objective, "student_rollout_replay", "dmd2")
            _reject_unknown(
                replay,
                {
                    "enabled", "initial_blocking", "refresh_generator_updates",
                    "capacity_replans", "device", "fps", "base_reset_seed",
                    "max_episode_steps",
                },
                "dmd2.student_rollout_replay",
            )
            if not bool(replay.get("enabled", False)):
                raise ConfigError("configured DMD2 student replay must be enabled")
            if not bool(replay.get("initial_blocking", False)):
                raise ConfigError("DMD2 student replay must collect a balanced initial rollout")
            if int(replay.get("refresh_generator_updates", 0)) != 500:
                raise ConfigError("DMD2 student replay refresh must be every 500 generator updates")
            if int(replay.get("capacity_replans", 0)) < 40:
                raise ConfigError("DMD2 student replay capacity must be at least 40 replans")
            if int(replay.get("max_episode_steps", 0)) < 1:
                raise ConfigError("DMD2 student replay max_episode_steps must be positive")
        if objective.get("fake_score_variant") != "pi05_clone" and "ablation" not in str(
            config.get("classification", "")
        ).lower():
            raise ConfigError(f"baseline {method} must use fake_score_variant=pi05_clone")
        if "fake_score" in objective and objective.get("fake_score_variant") != "lightweight":
            raise ConfigError(
                f"{section_name}.fake_score is only valid for the explicitly lightweight ablation"
            )
        discriminator = _require(objective, "discriminator", section_name)
        if discriminator.get("variant") not in {"pi05_intermediate_features", "cached_vla_features"}:
            raise ConfigError(
                f"{section_name}.discriminator.variant must explicitly select paper or cached VLA features"
            )
        if (
            discriminator["variant"] != "pi05_intermediate_features"
            and "ablation" not in str(config.get("classification", "")).lower()
        ):
            raise ConfigError(f"baseline {method} must use the paper intermediate-feature discriminator")
        if discriminator["variant"] == "pi05_intermediate_features":
            _reject_unknown(
                discriminator,
                {"variant", "feature_source", "selected_layers", "hidden_dim", "time_embedding_dim"},
                f"{section_name}.discriminator",
            )
            expected_source = "fake_score_features" if method == "dmd2_flow" else "teacher_features"
            if discriminator.get("feature_source") != expected_source:
                raise ConfigError(
                    f"{section_name} paper discriminator requires feature_source={expected_source}"
                )
            if method == "dmd2_flow" and float(objective.get("guidance_classifier_weight", 0.0)) <= 0:
                raise ConfigError("dmd2.guidance_classifier_weight must be positive")
            if method == "dmd2_flow" and "discriminator_updates_per_generator" in objective:
                raise ConfigError(
                    "dmd2 fake-score-feature guidance jointly updates the classifier; "
                    "discriminator_updates_per_generator is invalid"
                )
            if method == "tmd_stage2" and int(
                objective.get("discriminator_updates_per_generator", 0)
            ) < 1:
                raise ConfigError("stage2.discriminator_updates_per_generator must be positive")
            selected = [int(value) for value in discriminator.get("selected_layers", [])]
            if not selected or len(selected) != len(set(selected)) or min(selected) < 0:
                raise ConfigError(f"{section_name}.discriminator.selected_layers must be nonempty and unique")
        else:
            _reject_unknown(
                discriminator,
                {"variant", "feature_source", "num_tasks", "model_dim", "layers", "heads"},
                f"{section_name}.discriminator",
            )
            if discriminator.get("feature_source") != "cached_vla_features":
                raise ConfigError(f"{section_name} cached discriminator must identify cached_vla_features")
            if int(objective.get("discriminator_updates_per_generator", 0)) < 1:
                raise ConfigError(
                    f"{section_name}.discriminator_updates_per_generator must be positive"
                )
            if "guidance_classifier_weight" in objective:
                raise ConfigError(
                    f"{section_name}.guidance_classifier_weight is only valid for fake-score features"
                )
        for name in ("vsd_time", "gan_time", "fake_score_time"):
            timing = _require(objective, name, section_name)
            minimum = float(_require(timing, "minimum_time", f"{section_name}.{name}"))
            maximum = float(_require(timing, "maximum_time", f"{section_name}.{name}"))
            gamma = float(_require(timing, "time_shift_gamma", f"{section_name}.{name}"))
            if not 0 <= minimum < maximum <= 1 or gamma < 1:
                raise ConfigError(f"invalid {section_name}.{name} time range/shift")

    if method == "collect_student":
        _all_libero_tasks(_require(config["collection"], "benchmark", "collection"), "collection.benchmark")
        _reject_unknown(
            config["policy"],
            {"method", "device", "checkpoint", "checkpoint_sha256"},
            "policy",
        )

    if method == "occupancy_tmd":
        _reject_unknown(
            config["occupancy"],
            {
                "discriminator_checkpoint", "discriminator_checkpoint_sha256", "minimum_weight",
                "maximum_weight", "temperature", "weight_sampler_outer_steps",
                "weight_sampler_inner_steps", "rollout_mode",
            },
            "occupancy",
        )

    if method == "occupancy_discriminator":
        occupancy = _require(config, "occupancy_discriminator", "config")
        allowed = {
            "variant", "window_length", "minimum_time", "maximum_time", "time_shift_gamma",
            "selected_layers", "hidden_dim", "time_embedding_dim", "num_tasks", "model_dim",
            "layers", "heads",
        }
        _reject_unknown(occupancy, allowed, "occupancy_discriminator")
        if occupancy.get("variant") not in {"pi05_intermediate_features", "cached_vla_features"}:
            raise ConfigError("occupancy_discriminator.variant must explicitly select a feature source")
        configured = set(int(value) for value in config["rollouts"].get("configured_task_indices", []))
        if configured != set(range(40)):
            raise ConfigError("occupancy rollouts must configure all 40 global task indices")
        minimum = float(_require(occupancy, "minimum_time", "occupancy_discriminator"))
        maximum = float(_require(occupancy, "maximum_time", "occupancy_discriminator"))
        gamma = float(_require(occupancy, "time_shift_gamma", "occupancy_discriminator"))
        if not 0 <= minimum < maximum <= 1 or gamma < 1:
            raise ConfigError("invalid occupancy discriminator time range/shift")

    if method in {"evaluate_libero", "evaluate_libero_plus"}:
        policy = _require(config, "policy", "config")
        policy_method = str(_require(policy, "method", "policy"))
        if policy_method not in {
            "pi05", "smolvla", "flow_sft", "tmd_stage1", "dmd2_flow", "tmd_stage2", "occupancy_tmd"
        }:
            raise ConfigError(f"unknown evaluation policy method: {policy_method}")
        if policy_method == "pi05":
            _reject_unknown(policy, {"method", "device", "num_steps"}, "policy")
        elif policy_method == "smolvla":
            _reject_unknown(
                policy, {"method", "device", "sampler_mode", "num_steps", "classification"}, "policy"
            )
        else:
            sampling_override_fields = (
                {"outer_steps", "inner_steps"}
                if policy_method == "flow_sft"
                else {"outer_steps"}
                if policy_method == "dmd2_flow"
                else set()
            )
            _reject_unknown(
                policy,
                {"method", "device", "checkpoint", "checkpoint_sha256"}
                | sampling_override_fields,
                "policy",
            )
        if policy_method == "smolvla":
            mode = str(_require(policy, "sampler_mode", "policy"))
            if mode == "official" and ({"num_steps", "outer_steps"} & set(policy)):
                raise ConfigError("official SmolVLA evaluation forbids step overrides")
            if mode == "override" and "ablation" not in str(policy.get("classification", "")).lower():
                raise ConfigError("SmolVLA override mode must be classified as an ablation")

    if method == "evaluate_libero_plus":
        evaluation = _require(config, "evaluation", "config")
        _reject_unknown(
            evaluation,
            {
                "fps", "suites", "task_ids", "reset_seeds", "suite_max_episode_steps",
                "expected_total_tasks", "synchronize_cuda", "control_mode", "hard_reset",
                "sample_per_category", "sample_seed",
            },
            "evaluation",
        )
        suites = [str(value) for value in _require(evaluation, "suites", "evaluation")]
        expected_suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
        if not suites or len(suites) != len(set(suites)) or any(
            suite not in expected_suites for suite in suites
        ):
            raise ConfigError(
                "LIBERO-Plus evaluation.suites must be a unique nonempty subset of the four suites"
            )
        if int(_require(evaluation, "expected_total_tasks", "evaluation")) != 10_030:
            raise ConfigError("the full four-suite LIBERO-Plus contract contains exactly 10,030 tasks")
        task_ids = evaluation.get("task_ids")
        sample_per_category = evaluation.get("sample_per_category")
        if task_ids is not None and sample_per_category is not None:
            raise ConfigError(
                "LIBERO-Plus task_ids and sample_per_category are mutually exclusive"
            )
        if task_ids is not None:
            normalized_ids = [int(value) for value in task_ids]
            if (
                not normalized_ids
                or len(normalized_ids) != len(set(normalized_ids))
                or min(normalized_ids) < 0
            ):
                raise ConfigError("LIBERO-Plus task_ids must be unique nonnegative integers")
        if sample_per_category is not None and int(sample_per_category) < 1:
            raise ConfigError("LIBERO-Plus sample_per_category must be positive")
        if int(evaluation.get("sample_seed", 0)) < 0:
            raise ConfigError("LIBERO-Plus sample_seed must be nonnegative")
        reset_seeds = [int(value) for value in _require(evaluation, "reset_seeds", "evaluation")]
        if not reset_seeds or len(reset_seeds) != len(set(reset_seeds)) or min(reset_seeds) < 0:
            raise ConfigError("LIBERO-Plus reset_seeds must be unique nonnegative integers")
        episode_steps = _require(evaluation, "suite_max_episode_steps", "evaluation")
        if set(episode_steps) != set(expected_suites) or min(
            int(value) for value in episode_steps.values()
        ) < 1:
            raise ConfigError("LIBERO-Plus suite_max_episode_steps must cover all four suites")
        if int(evaluation.get("fps", 0)) < 1:
            raise ConfigError("LIBERO-Plus evaluation.fps must be positive")
        if evaluation.get("control_mode", "relative") not in {"relative", "absolute"}:
            raise ConfigError("LIBERO-Plus control_mode must be relative or absolute")


def load_config(path: str | Path, *, expected_method: str | None = None) -> dict[str, Any]:
    path = Path(path)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    config = copy.deepcopy(value)
    config["_config_path"] = str(path.resolve())
    validate_config(config, expected_method=expected_method)
    return config


def save_resolved_config(config: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


__all__ = [
    "ConfigError",
    "IMMUTABLE_REVISION",
    "LEROBOT_VERSION",
    "PROJECT_ROOT",
    "load_config",
    "project_path",
    "save_resolved_config",
    "validate_config",
]
