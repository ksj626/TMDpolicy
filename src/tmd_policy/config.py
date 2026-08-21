"""Strict configuration loading for the DMD2-only production surface."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

from tmd_policy.libero_protocol import LIBERO_SUITE_MAX_EPISODE_STEPS, validate_suite_max_episode_steps

LEROBOT_VERSION = "0.6.1"
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_METHODS = {
    "data_build_expert",
    "pi05_flow_parity",
    "dmd2_flow",
    "collect_student",
    "evaluate_libero",
    "evaluate_libero_plus",
}
SUPPORTED_POLICIES = {"pi05", "smolvla", "dmd2_flow"}


class ConfigError(ValueError):
    """Raised before models or datasets are loaded when a run is ambiguous."""


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required configuration field: {context}.{key}")
    return mapping[key]


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigError(f"unknown or deprecated configuration fields in {context}: {unknown}")


def _positive_int(value: Any, field: str) -> int:
    result = int(value)
    if result < 1:
        raise ConfigError(f"{field} must be positive")
    return result


def _revision(value: Any, field: str) -> str:
    result = str(value)
    if not IMMUTABLE_REVISION.fullmatch(result):
        raise ConfigError(f"{field} must be an immutable 40-character Hub commit, got {result!r}")
    return result


def _suite_steps(value: Any, context: str) -> dict[str, int]:
    try:
        return validate_suite_max_episode_steps(value, context=context)
    except (TypeError, ValueError) as error:
        raise ConfigError(str(error)) from error


def _libero_tasks(benchmark: Any, context: str, *, require_all: bool) -> None:
    if not isinstance(benchmark, list) or not benchmark:
        raise ConfigError(f"{context} must be a nonempty suite/task list")
    actual: dict[str, set[int]] = {}
    for entry in benchmark:
        suite = str(entry.get("suite"))
        ids = [int(value) for value in entry.get("task_ids", ())]
        if suite not in LIBERO_SUITE_MAX_EPISODE_STEPS or suite in actual:
            raise ConfigError(f"{context} contains an unknown or repeated suite: {suite}")
        if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(range(10)):
            raise ConfigError(f"{context} task_ids must be unique values in [0,9]")
        actual[suite] = set(ids)
    if require_all:
        expected = {suite: set(range(10)) for suite in LIBERO_SUITE_MAX_EPISODE_STEPS}
        if actual != expected:
            raise ConfigError(f"{context} must cover each of the four suites and all 40 tasks exactly")


def _validate_assets(config: dict[str, Any]) -> None:
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
        _revision(asset["processor_revision"], f"models.{name}.processor_revision")

    dataset = _require(config, "dataset", "config")
    _reject_unknown(
        dataset,
        {"id", "revision", "cache", "manifest", "validation_fraction", "test_fraction", "split_seed", "download_videos", "video_backend"},
        "dataset",
    )
    if dataset["id"] != "lerobot/libero":
        raise ConfigError("the canonical expert dataset must be lerobot/libero")
    _revision(dataset["revision"], "dataset.revision")
    validation = float(dataset.get("validation_fraction", 0.1))
    test = float(dataset.get("test_fraction", 0.1))
    if min(validation, test) <= 0 or validation + test >= 1:
        raise ConfigError("validation/test fractions must be positive and sum to less than one")

    horizons = _require(config, "horizons", "config")
    _reject_unknown(horizons, {"prediction", "execution"}, "horizons")
    if int(horizons["prediction"]) != 50:
        raise ConfigError("LIBERO prediction horizon is fixed at 50")
    if not 1 <= int(horizons["execution"]) <= 50:
        raise ConfigError("horizons.execution must be in [1, 50]")
    _reject_unknown(_require(config, "output", "config"), {"directory"}, "output")


def _validate_policy(policy: dict[str, Any]) -> None:
    method = str(_require(policy, "method", "policy"))
    if method not in SUPPORTED_POLICIES:
        raise ConfigError(f"unsupported evaluation policy method: {method!r}")
    common = {"method", "device"}
    if method == "pi05":
        allowed = common | {"num_steps"}
        _positive_int(policy.get("num_steps", 10), "policy.num_steps")
    elif method == "smolvla":
        allowed = common | {"sampler_mode", "num_steps", "classification"}
        mode = str(policy.get("sampler_mode", "official"))
        if mode not in {"official", "override"}:
            raise ConfigError("policy.sampler_mode must be official or override")
        if mode == "official" and ("num_steps" in policy or "classification" in policy):
            raise ConfigError("official SmolVLA forbids sampler overrides and ablation labels")
        if mode == "override":
            _positive_int(_require(policy, "num_steps", "policy"), "policy.num_steps")
            if "ablation" not in str(_require(policy, "classification", "policy")).lower():
                raise ConfigError("SmolVLA sampler overrides must be explicitly classified as an ablation")
    else:
        allowed = common | {"checkpoint", "checkpoint_sha256", "outer_steps"}
        _require(policy, "checkpoint", "policy")
        _require(policy, "checkpoint_sha256", "policy")
        if "outer_steps" in policy:
            _positive_int(policy["outer_steps"], "policy.outer_steps")
    _reject_unknown(policy, allowed, "policy")


def _validate_dmd2(config: dict[str, Any]) -> None:
    training = _require(config, "training", "config")
    _reject_unknown(
        training,
        {"seed", "device", "batch_size", "num_workers", "mixed_precision", "gradient_accumulation", "gradient_clip_norm", "learning_rate", "weight_decay", "betas", "epsilon", "warmup_steps", "minimum_lr_scale", "max_steps", "validation_interval", "validation_batches", "checkpoint_interval", "inference_checkpoint_interval", "diagnostics_interval"},
        "training",
    )
    for field in ("batch_size", "gradient_accumulation", "max_steps", "validation_interval", "validation_batches", "checkpoint_interval", "inference_checkpoint_interval", "diagnostics_interval"):
        _positive_int(training[field], f"training.{field}")
    if training["mixed_precision"] not in {"no", "fp16", "bf16"}:
        raise ConfigError("training.mixed_precision must be no, fp16, or bf16")

    dmd2 = _require(config, "dmd2", "config")
    _reject_unknown(
        dmd2,
        {"student_fine_tuning", "fake_score_variant", "fake_updates_per_generator", "guidance_classifier_weight", "student_training_mode", "discrete_outer_steps", "student_time_shift_gamma", "generator_learning_rate", "fake_score_learning_rate", "discriminator_learning_rate", "gan_weight", "vsd_normalization", "vsd_normalization_epsilon", "teacher_device", "teacher_dtype", "fake_score_device", "student_rollout_replay", "vsd_time", "gan_time", "fake_score_time", "discriminator", "resource_estimate"},
        "dmd2",
    )
    required_values = {
        "student_fine_tuning": "action_expert",
        "fake_score_variant": "pi05_clone",
        "student_training_mode": "backward_simulation_denoise_renoise",
        "vsd_normalization": "dmd2_teacher_residual_mean_abs",
    }
    for field, expected in required_values.items():
        if dmd2.get(field) != expected:
            raise ConfigError(f"dmd2.{field} must be {expected!r}")
    if int(dmd2.get("fake_updates_per_generator", 0)) != 5:
        raise ConfigError("dmd2.fake_updates_per_generator must be 5")
    _positive_int(dmd2["discrete_outer_steps"], "dmd2.discrete_outer_steps")
    if float(dmd2["student_time_shift_gamma"]) < 1:
        raise ConfigError("dmd2.student_time_shift_gamma must be at least one")
    for name in ("vsd_time", "gan_time", "fake_score_time"):
        section = _require(dmd2, name, "dmd2")
        _reject_unknown(section, {"minimum_time", "maximum_time", "time_shift_gamma"}, f"dmd2.{name}")
        minimum, maximum, gamma = float(section["minimum_time"]), float(section["maximum_time"]), float(section["time_shift_gamma"])
        if not 0 < minimum < maximum <= 1 or gamma < 1:
            raise ConfigError(f"invalid dmd2.{name} time range/shift")
    discriminator = _require(dmd2, "discriminator", "dmd2")
    _reject_unknown(discriminator, {"variant", "feature_source", "selected_layers", "hidden_dim", "time_embedding_dim"}, "dmd2.discriminator")
    if discriminator.get("variant") != "pi05_intermediate_features":
        raise ConfigError("dmd2.discriminator.variant must be 'pi05_intermediate_features'")
    if discriminator.get("feature_source") != "fake_score_features":
        raise ConfigError("dmd2.discriminator.feature_source must be 'fake_score_features'")
    if not discriminator.get("selected_layers"):
        raise ConfigError("dmd2.discriminator.selected_layers must be nonempty")

    replay = _require(dmd2, "student_rollout_replay", "dmd2")
    _reject_unknown(replay, {"enabled", "initial_blocking", "refresh_generator_updates", "capacity_replans", "device", "devices", "fps", "base_reset_seed", "suite_max_episode_steps", "batch_size", "validation_videos"}, "dmd2.student_rollout_replay")
    if not replay.get("enabled") or not replay.get("initial_blocking"):
        raise ConfigError("DMD2 requires enabled balanced initial student replay")
    if _positive_int(replay["capacity_replans"], "dmd2.student_rollout_replay.capacity_replans") < 40:
        raise ConfigError("DMD2 student replay capacity must be at least 40 replans")
    _positive_int(replay["refresh_generator_updates"], "dmd2.student_rollout_replay.refresh_generator_updates")
    _positive_int(replay["batch_size"], "dmd2.student_rollout_replay.batch_size")
    devices = [str(value) for value in replay.get("devices", [replay["device"]])]
    if not devices or len(devices) != len(set(devices)):
        raise ConfigError("DMD2 student replay devices must be unique and nonempty")
    _suite_steps(replay["suite_max_episode_steps"], "dmd2.student_rollout_replay.suite_max_episode_steps")
    videos = _require(replay, "validation_videos", "dmd2.student_rollout_replay")
    _reject_unknown(videos, {"enabled", "task_ids", "simulator_seed", "init_state_index", "camera"}, "dmd2.student_rollout_replay.validation_videos")
    preflight = _require(config, "preflight", "config")
    _reject_unknown(preflight, {"minimum_total_memory_gib"}, "preflight")


def _validate_collection(config: dict[str, Any]) -> None:
    _validate_policy(config["policy"])
    if config["policy"]["method"] != "dmd2_flow":
        raise ConfigError("student rollout collection requires policy.method='dmd2_flow'")
    collection = _require(config, "collection", "config")
    _reject_unknown(collection, {"fps", "devices", "batch_size", "benchmark", "train_reset_seeds", "train_init_state_indices", "validation_reset_seeds", "validation_init_state_indices", "validation_task_ids", "save_validation_videos", "video_camera", "suite_max_episode_steps", "collection_round"}, "collection")
    _libero_tasks(collection["benchmark"], "collection.benchmark", require_all=True)
    _suite_steps(collection["suite_max_episode_steps"], "collection.suite_max_episode_steps")
    _positive_int(collection["batch_size"], "collection.batch_size")


def _validate_evaluation(config: dict[str, Any], *, plus: bool) -> None:
    _validate_policy(config["policy"])
    evaluation = _require(config, "evaluation", "config")
    if plus:
        allowed = {"fps", "suites", "task_ids", "sample_per_category", "sample_seed", "batch_size", "save_videos", "video_camera", "reset_seeds", "suite_max_episode_steps", "expected_total_tasks", "synchronize_cuda", "control_mode", "hard_reset", "devices"}
        _reject_unknown(evaluation, allowed, "evaluation")
        suites = [str(value) for value in evaluation["suites"]]
        if not suites or len(suites) != len(set(suites)) or not set(suites) <= set(LIBERO_SUITE_MAX_EPISODE_STEPS):
            raise ConfigError("evaluation.suites must be unique known LIBERO suites")
        if evaluation.get("task_ids") is not None and evaluation.get("sample_per_category") is not None:
            raise ConfigError("evaluation.task_ids and sample_per_category are mutually exclusive")
        _positive_int(evaluation["batch_size"], "evaluation.batch_size")
        _positive_int(evaluation["expected_total_tasks"], "evaluation.expected_total_tasks")
    else:
        _reject_unknown(evaluation, {"fps", "benchmark", "reset_seeds", "suite_max_episode_steps", "save_rollouts", "synchronize_cuda"}, "evaluation")
        _libero_tasks(evaluation["benchmark"], "evaluation.benchmark", require_all=False)
    _suite_steps(evaluation["suite_max_episode_steps"], "evaluation.suite_max_episode_steps")
    seeds = [int(value) for value in evaluation["reset_seeds"]]
    if not seeds or len(seeds) != len(set(seeds)) or min(seeds) < 0:
        raise ConfigError("evaluation.reset_seeds must be unique nonnegative integers")


def validate_config(config: dict[str, Any], *, expected_method: str | None = None) -> None:
    method = str(_require(config, "method", "config"))
    if method not in SUPPORTED_METHODS:
        raise ConfigError(f"unknown production method: {method!r}")
    if expected_method is not None and method != expected_method:
        raise ConfigError(f"command expects method={expected_method!r}, config declares {method!r}")
    root_fields = {
        "data_build_expert": {"method", "backend", "models", "dataset", "horizons", "output"},
        "pi05_flow_parity": {"method", "classification", "backend", "models", "dataset", "horizons", "parity", "output"},
        "dmd2_flow": {"method", "classification", "backend", "models", "dataset", "horizons", "training", "dmd2", "preflight", "output"},
        "collect_student": {"method", "classification", "backend", "models", "dataset", "horizons", "policy", "collection", "output"},
        "evaluate_libero": {"method", "classification", "backend", "models", "dataset", "horizons", "policy", "evaluation", "output"},
        "evaluate_libero_plus": {"method", "classification", "backend", "models", "dataset", "horizons", "policy", "evaluation", "output"},
    }
    _reject_unknown(config, root_fields[method] | {"_config_path"}, "config")
    _validate_assets(config)
    if method == "dmd2_flow":
        _validate_dmd2(config)
    elif method == "collect_student":
        _validate_collection(config)
    elif method == "evaluate_libero":
        _validate_evaluation(config, plus=False)
    elif method == "evaluate_libero_plus":
        _validate_evaluation(config, plus=True)
    elif method == "pi05_flow_parity":
        parity = config["parity"]
        _reject_unknown(parity, {"device", "dtype", "num_steps", "noise_seed", "minimum_score_time"}, "parity")
        _positive_int(parity["num_steps"], "parity.num_steps")


def load_config(path: str | Path, *, expected_method: str | None = None) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"configuration root must be a mapping: {resolved}")
    config = copy.deepcopy(config)
    config["_config_path"] = str(resolved)
    validate_config(config, expected_method=expected_method)
    return config


def save_resolved_config(config: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


__all__ = ["ConfigError", "PROJECT_ROOT", "load_config", "project_path", "save_resolved_config", "validate_config"]
