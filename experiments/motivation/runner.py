from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tmd_policy.evaluation.metrics import (
    binary_metrics,
    bootstrap_episode_statistic,
    prefix_discriminator_report,
)
from tmd_policy.models.discriminator import CausalPathDiscriminator, DiscriminatorVariant
from tmd_policy.training.discriminator import discriminator_loss

from .plots import plot_m0, plot_m1, plot_m2, plot_m3, plot_m4, plot_m5
from .synthetic import PathBatch, generate_paths, make_splits

SYNTHETIC_CHECKPOINT = "synthetic-diagnostic/no-policy-checkpoint"
SYNTHETIC_REVISION = "synthetic-v1"
SPLIT_COUNTS = (256, 128, 512)
TRAINING_STEPS = 60


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metadata(experiment: str, episodes: int, seeds: list[int], data_file: str) -> str:
    return (
        f"{experiment} | SYNTHETIC | tasks=0,1,2,3 | checkpoint={SYNTHETIC_CHECKPOINT} "
        f"revision={SYNTHETIC_REVISION} | episodes={episodes} | "
        f"split={SPLIT_COUNTS} | seeds={seeds} | data={data_file}"
    )


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fit_discriminator(
    positive: PathBatch,
    negative: PathBatch,
    *,
    variant: DiscriminatorVariant,
    seed: int,
    model_kwargs: dict[str, Any] | None = None,
    training_steps: int = TRAINING_STEPS,
) -> CausalPathDiscriminator:
    torch.manual_seed(seed)
    settings = {
        "num_tasks": 4,
        "model_dim": 32,
        "num_layers": 1,
        "num_heads": 4,
        "feedforward_dim": 64,
        "dropout": 0.05,
    }
    settings.update(model_kwargs or {})
    model = CausalPathDiscriminator(
        state_dim=8,
        action_dim=7,
        execution_horizon=10,
        variant=variant,
        **settings,
    )
    model.normalizer.fit(
        torch.cat((positive.states, negative.states)),
        torch.cat((positive.actions, negative.actions)),
        torch.cat((positive.valid, negative.valid)),
        split="train",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(seed + 1)
    batch_size = min(64, len(positive), len(negative))
    for _ in range(training_steps):
        positive_indices = torch.randint(len(positive), (batch_size,), generator=generator)
        negative_indices = torch.randint(len(negative), (batch_size,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = discriminator_loss(
            model,
            positive.select(positive_indices).model_dict(),
            negative.select(negative_indices).model_dict(),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    return model.eval()


@torch.no_grad()
def _logits(model: CausalPathDiscriminator, batch: PathBatch) -> np.ndarray:
    return model(batch.states, batch.actions, batch.task_ids, batch.valid).cpu().numpy()


def _final(logits: np.ndarray, valid: torch.Tensor) -> np.ndarray:
    if logits.shape[1] == 1:
        return logits[:, 0]
    last = valid.sum(dim=1).numpy().clip(min=1) - 1
    return logits[np.arange(len(logits)), last]


def _curve(labels: np.ndarray, logits: np.ndarray) -> dict[str, list[float]]:
    order = np.argsort(-logits, kind="mergesort")
    sorted_labels = labels[order]
    positives = max(int(sorted_labels.sum()), 1)
    negatives = max(len(labels) - int(sorted_labels.sum()), 1)
    true_positive = np.cumsum(sorted_labels)
    false_positive = np.cumsum(1 - sorted_labels)
    precision = true_positive / np.arange(1, len(labels) + 1)
    recall = true_positive / positives
    return {
        "fpr": np.concatenate(([0.0], false_positive / negatives)).tolist(),
        "tpr": np.concatenate(([0.0], true_positive / positives)).tolist(),
        "precision": np.concatenate(([1.0], precision)).tolist(),
        "recall": np.concatenate(([0.0], recall)).tolist(),
    }


def _calibration(labels: np.ndarray, logits: np.ndarray, bins: int = 10) -> dict[str, list[float]]:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    confidence: list[float] = []
    accuracy: list[float] = []
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        selected = (probabilities >= low) & (probabilities <= high if high == 1 else probabilities < high)
        if selected.any():
            confidence.append(float(probabilities[selected].mean()))
            accuracy.append(float(labels[selected].mean()))
    return {"confidence": confidence, "accuracy": accuracy}


def _comparison(
    model: CausalPathDiscriminator,
    positive: PathBatch,
    negative: PathBatch,
) -> dict[str, Any]:
    positive_logits = _final(_logits(model, positive), positive.valid)
    negative_logits = _final(_logits(model, negative), negative.valid)
    labels = np.concatenate((np.ones(len(positive)), np.zeros(len(negative)))).astype(int)
    scores = np.concatenate((positive_logits, negative_logits))
    tasks = np.concatenate((positive.task_ids.numpy(), negative.task_ids.numpy()))
    per_task: dict[str, float] = {}
    for task in np.unique(tasks):
        selected = tasks == task
        per_task[str(int(task))] = binary_metrics(labels[selected], scores[selected])["roc_auc"]
    return {
        "metrics": binary_metrics(labels, scores),
        "curve": _curve(labels, scores),
        "calibration": _calibration(labels, scores),
        "per_task_auc": per_task,
        "positive_logits": positive_logits,
        "negative_logits": negative_logits,
    }


def run_m0(
    output_root: Path,
    *,
    seed: int = 101,
    model_kwargs: dict[str, Any] | None = None,
    training_steps: int = TRAINING_STEPS,
) -> dict[str, Any]:
    output = output_root / "M0"
    output.mkdir(parents=True, exist_ok=True)
    expert_a = make_splits(SPLIT_COUNTS, seed=seed + 10, domain="expert")
    expert_b = make_splits(SPLIT_COUNTS, seed=seed + 20, domain="expert")
    current_a = make_splits(SPLIT_COUNTS, seed=seed + 30, domain="current")
    current_b = make_splits(SPLIT_COUNTS, seed=seed + 40, domain="current")
    definitions = {
        "expert_A_vs_expert_B": (expert_a, expert_b, seed + 1),
        "current_A_vs_current_B": (current_a, current_b, seed + 2),
        "expert_vs_current": (expert_a, current_a, seed + 3),
    }
    comparisons: dict[str, Any] = {}
    raw: dict[str, np.ndarray] = {}
    for name, (positive, negative, model_seed) in definitions.items():
        model = _fit_discriminator(
            positive["train"],
            negative["train"],
            variant=DiscriminatorVariant.PREFIX,
            seed=model_seed,
            model_kwargs=model_kwargs,
            training_steps=training_steps,
        )
        result = _comparison(model, positive["test"], negative["test"])
        comparisons[name] = result
        raw[f"{name}_positive_logits"] = np.asarray(result["positive_logits"])
        raw[f"{name}_negative_logits"] = np.asarray(result["negative_logits"])
    control_deviation = max(
        abs(comparisons["expert_A_vs_expert_B"]["metrics"]["roc_auc"] - 0.5),
        abs(comparisons["current_A_vs_current_B"]["metrics"]["roc_auc"] - 0.5),
    )
    gate_passed = control_deviation <= 0.12
    raw_path = output / "raw_logits.npz"
    np.savez_compressed(raw_path, **raw)
    report = {
        "experiment": "M0",
        "data_label": "synthetic diagnostic",
        "checkpoint": SYNTHETIC_CHECKPOINT,
        "revision": SYNTHETIC_REVISION,
        "tasks": [0, 1, 2, 3],
        "split_episode_counts_per_source": list(SPLIT_COUNTS),
        "seeds": [seed + index for index in (1, 2, 3, 10, 20, 30, 40)],
        "comparisons": comparisons,
        "control_gate_threshold": 0.12,
        "maximum_control_auc_deviation_from_chance": control_deviation,
        "gate_passed": gate_passed,
        "raw_data": str(raw_path),
    }
    metrics_path = output / "metrics.json"
    _save_json(metrics_path, report)
    with (output / "comparison_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("comparison", "roc_auc", "pr_auc", "bce", "brier", "ece"))
        for name, result in comparisons.items():
            metrics = result["metrics"]
            writer.writerow(
                (name, metrics["roc_auc"], metrics["pr_auc"], metrics["bce"], metrics["brier"], metrics["ece"])
            )
    figures = plot_m0(
        _jsonable(report),
        output,
        _metadata("M0", 6 * SPLIT_COUNTS[2], report["seeds"], str(raw_path)),
    )
    report["figures"] = figures
    _save_json(metrics_path, report)
    return report


def _train_main_models(
    seed: int,
    model_kwargs: dict[str, Any] | None = None,
    training_steps: int = TRAINING_STEPS,
) -> tuple[dict[str, PathBatch], dict[str, PathBatch], dict[str, CausalPathDiscriminator]]:
    expert = make_splits(SPLIT_COUNTS, seed=seed + 10, domain="expert")
    current = make_splits(SPLIT_COUNTS, seed=seed + 20, domain="current")
    models = {
        variant.value: _fit_discriminator(
            expert["train"],
            current["train"],
            variant=variant,
            seed=seed + 100 + index,
            model_kwargs=model_kwargs,
            training_steps=training_steps,
        )
        for index, variant in enumerate(
            (DiscriminatorVariant.POINTWISE, DiscriminatorVariant.FINAL, DiscriminatorVariant.PREFIX)
        )
    }
    return expert, current, models


def run_m1(
    output_root: Path,
    expert: dict[str, PathBatch],
    current: dict[str, PathBatch],
    models: dict[str, CausalPathDiscriminator],
    *,
    seed: int,
) -> dict[str, Any]:
    output = output_root / "M1"
    output.mkdir(parents=True, exist_ok=True)
    variants = {
        name: _comparison(model, expert["test"], current["test"])
        for name, model in models.items()
    }
    prefix_model = models[DiscriminatorVariant.PREFIX.value]
    expert_prefix = _logits(prefix_model, expert["test"])
    current_prefix = _logits(prefix_model, current["test"])
    prefix_report = prefix_discriminator_report(
        expert_prefix,
        current_prefix,
        expert["test"].valid,
        current["test"].valid,
        expert_task_ids=expert["test"].task_ids,
        student_task_ids=current["test"].task_ids,
    )
    raw_path = output / "prefix_logits.npz"
    np.savez_compressed(
        raw_path,
        expert_prefix_logits=expert_prefix,
        current_prefix_logits=current_prefix,
        expert_task_ids=expert["test"].task_ids.numpy(),
        current_task_ids=current["test"].task_ids.numpy(),
    )
    prefix_data = {
        "expert_probability_mean": (1 / (1 + np.exp(-expert_prefix))).mean(0),
        "current_probability_mean": (1 / (1 + np.exp(-current_prefix))).mean(0),
        "expert_final_logits": expert_prefix[:, -1],
        "current_final_logits": current_prefix[:, -1],
    }
    report = {
        "experiment": "M1",
        "data_label": "synthetic diagnostic; not LIBERO occupancy evidence",
        "checkpoint": SYNTHETIC_CHECKPOINT,
        "revision": SYNTHETIC_REVISION,
        "tasks": [0, 1, 2, 3],
        "split_episode_counts_per_source": list(SPLIT_COUNTS),
        "seeds": [seed, seed + 100, seed + 101, seed + 102],
        "variants": variants,
        "prefix_report": prefix_report,
        "prefix": prefix_data,
        "raw_data": str(raw_path),
    }
    metrics_path = output / "metrics.json"
    _save_json(metrics_path, report)
    report["figures"] = plot_m1(
        _jsonable(report),
        output,
        _metadata("M1", 2 * SPLIT_COUNTS[2], report["seeds"], str(raw_path)),
    )
    _save_json(metrics_path, report)
    return report


def run_m2(
    output_root: Path,
    current: dict[str, PathBatch],
    prefix_model: CausalPathDiscriminator,
    *,
    seed: int,
) -> dict[str, Any]:
    output = output_root / "M2"
    output.mkdir(parents=True, exist_ok=True)
    test = current["test"]
    prefix = _logits(prefix_model, test)
    final = _final(prefix, test.valid)
    success = test.success.numpy().astype(int)
    stacked = np.stack((final, success), axis=1)

    def difference(values: np.ndarray) -> float:
        labels = values[:, 1].astype(bool)
        return float(values[labels, 0].mean() - values[~labels, 0].mean())

    by_task: dict[str, float] = {}
    for task in np.unique(test.task_ids.numpy()):
        selected = test.task_ids.numpy() == task
        by_task[str(int(task))] = float(np.corrcoef(final[selected], success[selected])[0, 1])
    raw_path = output / "episode_scores.npz"
    np.savez_compressed(
        raw_path,
        final_logits=final,
        success=success,
        task_ids=test.task_ids.numpy(),
        episode_ids=test.episode_ids.numpy(),
    )
    report = {
        "experiment": "M2",
        "data_label": "synthetic diagnostic; discriminator never trained on success labels",
        "success_prediction": binary_metrics(success, final),
        "success_minus_failure_logit": difference(stacked),
        "episode_bootstrap_ci": bootstrap_episode_statistic(
            stacked,
            statistic=difference,
            task_ids=test.task_ids.numpy(),
            resamples=500,
            seed=seed,
        ),
        "per_task_logit_success_correlation": by_task,
        "success": success,
        "final_logits": final,
        "raw_data": str(raw_path),
        "seeds": [seed],
    }
    metrics_path = output / "metrics.json"
    _save_json(metrics_path, report)
    report["figures"] = plot_m2(
        _jsonable(report),
        output,
        _metadata("M2", len(test), [seed], str(raw_path)),
    )
    _save_json(metrics_path, report)
    return report


def run_m3(
    output_root: Path,
    current: dict[str, PathBatch],
    prefix_model: CausalPathDiscriminator,
    *,
    seed: int,
) -> dict[str, Any]:
    output = output_root / "M3"
    output.mkdir(parents=True, exist_ok=True)
    test = current["test"]
    prefix = _logits(prefix_model, test)
    increments = np.diff(np.concatenate((np.zeros((len(test), 1)), prefix), axis=1), axis=1)
    action_norm = np.linalg.norm(test.actions[0].numpy(), axis=1)
    report_metrics = prefix_discriminator_report(
        prefix,
        prefix,
        test.valid,
        test.valid,
        student_success=test.success,
        student_failure_moments=test.failure_moments,
    )
    raw_path = output / "temporal_scores.npz"
    np.savez_compressed(
        raw_path,
        prefix_logits=prefix,
        increments=increments,
        success=test.success.numpy(),
        failure_moments=test.failure_moments.numpy(),
        actions=test.actions.numpy(),
        states=test.states.numpy(),
    )
    report = {
        "experiment": "M3",
        "data_label": "synthetic diagnostic",
        "interpretation": (
            "Each finite-calibration increment estimates a conditional log-ratio change; "
            "it is not asserted to be an exact reward."
        ),
        "metrics": report_metrics,
        "prefix_logits": prefix,
        "increments": increments,
        "success": test.success.numpy(),
        "example_action_norm": action_norm,
        "raw_data": str(raw_path),
        "seeds": [seed],
    }
    metrics_path = output / "metrics.json"
    _save_json(metrics_path, report)
    report["figures"] = plot_m3(
        _jsonable(report),
        output,
        _metadata("M3", len(test), [seed], str(raw_path)),
    )
    _save_json(metrics_path, report)
    return report


def run_m4(
    output_root: Path,
    prefix_model: CausalPathDiscriminator,
    *,
    seed: int,
) -> dict[str, Any]:
    output = output_root / "M4"
    output.mkdir(parents=True, exist_ok=True)
    standard = generate_paths(512, seed=seed, domain="current")
    perturbed = generate_paths(512, seed=seed, domain="perturbed")
    standard_logits = _final(_logits(prefix_model, standard), standard.valid)
    perturbed_logits = _final(_logits(prefix_model, perturbed), perturbed.valid)
    standard_features = np.concatenate(
        (standard.states.numpy().mean(axis=1), standard.actions.numpy().mean(axis=1)), axis=1
    )
    perturbed_features = np.concatenate(
        (perturbed.states.numpy().mean(axis=1), perturbed.actions.numpy().mean(axis=1)), axis=1
    )
    support_distance = float(np.linalg.norm(standard_features.mean(0) - perturbed_features.mean(0)))
    raw_path = output / "perturbation_scores.npz"
    np.savez_compressed(
        raw_path,
        standard_final_logits=standard_logits,
        perturbed_final_logits=perturbed_logits,
        standard_success=standard.success.numpy(),
        perturbed_success=perturbed.success.numpy(),
    )
    report = {
        "experiment": "M4",
        "data_label": "synthetic perturbation diagnostic; not a robot result",
        "standard_success": float(standard.success.float().mean()),
        "perturbed_success": float(perturbed.success.float().mean()),
        "standard_final_logits": standard_logits,
        "perturbed_final_logits": perturbed_logits,
        "mean_logit_shift": float(perturbed_logits.mean() - standard_logits.mean()),
        "state_action_support_distance": support_distance,
        "raw_data": str(raw_path),
        "seeds": [seed],
    }
    metrics_path = output / "metrics.json"
    _save_json(metrics_path, report)
    report["figures"] = plot_m4(
        _jsonable(report),
        output,
        _metadata("M4", 1024, [seed], str(raw_path)),
    )
    _save_json(metrics_path, report)
    return report


def run_m5(
    output_root: Path,
    prefix_model: CausalPathDiscriminator,
    *,
    seed: int,
) -> dict[str, Any]:
    output = output_root / "M5"
    output.mkdir(parents=True, exist_ok=True)
    original = generate_paths(512, seed=seed, domain="current")
    coarse = generate_paths(512, seed=seed, domain="coarse")
    original_logits = _final(_logits(prefix_model, original), original.valid)
    coarse_logits = _final(_logits(prefix_model, coarse), coarse.valid)
    action_mae = float((original.actions - coarse.actions).abs().mean())
    diversity = float(coarse.actions.std(dim=0, unbiased=False).mean())
    raw_path = output / "coarse_sampling_scores.npz"
    np.savez_compressed(
        raw_path,
        original_actions=original.actions.numpy(),
        coarse_actions=coarse.actions.numpy(),
        original_final_logits=original_logits,
        coarse_final_logits=coarse_logits,
        original_success=original.success.numpy(),
        coarse_success=coarse.success.numpy(),
    )
    report = {
        "experiment": "M5",
        "data_label": "synthetic coarse-sampling proxy; not SmolVLA or robot evidence",
        "success_rates": [
            float(original.success.float().mean()),
            float(coarse.success.float().mean()),
        ],
        "action_mae": action_mae,
        "coarse_diversity": diversity,
        "original_final_logits": original_logits,
        "coarse_final_logits": coarse_logits,
        "prefix_score_shift": float(coarse_logits.mean() - original_logits.mean()),
        "latency_s": None,
        "latency_reason": "synthetic paths do not execute SmolVLA; real latency belongs to B0/B1",
        "raw_data": str(raw_path),
        "seeds": [seed],
    }
    metrics_path = output / "metrics.json"
    _save_json(metrics_path, report)
    report["figures"] = plot_m5(
        _jsonable(report),
        output,
        _metadata("M5", 1024, [seed], str(raw_path)),
    )
    _save_json(metrics_path, report)
    return report


def run_experiments(
    output_root: Path,
    experiments: list[str],
    *,
    seed: int = 101,
    model_kwargs: dict[str, Any] | None = None,
    training_steps: int = TRAINING_STEPS,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    requested = [name.upper() for name in experiments]
    unknown = sorted(set(requested) - {"M0", "M1", "M2", "M3", "M4", "M5"})
    if unknown:
        raise ValueError(f"unknown motivation experiments: {unknown}")
    results: dict[str, Any] = {}
    if "M0" in requested:
        results["M0"] = run_m0(
            output_root,
            seed=seed,
            model_kwargs=model_kwargs,
            training_steps=training_steps,
        )
    if any(name != "M0" for name in requested):
        gate_path = output_root / "M0" / "metrics.json"
        if not gate_path.is_file():
            raise RuntimeError("run M0 first; later experiments require its saved source-control gate")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if not gate.get("gate_passed"):
            raise RuntimeError("M0 source controls are above the configured chance threshold; stopping")
        expert, current, models = _train_main_models(
            seed + 1_000, model_kwargs=model_kwargs, training_steps=training_steps
        )
        prefix = models[DiscriminatorVariant.PREFIX.value]
        if "M1" in requested:
            results["M1"] = run_m1(output_root, expert, current, models, seed=seed + 1_000)
        if "M2" in requested:
            results["M2"] = run_m2(output_root, current, prefix, seed=seed + 2_000)
        if "M3" in requested:
            results["M3"] = run_m3(output_root, current, prefix, seed=seed + 3_000)
        if "M4" in requested:
            results["M4"] = run_m4(output_root, prefix, seed=seed + 4_000)
        if "M5" in requested:
            results["M5"] = run_m5(output_root, prefix, seed=seed + 5_000)
    summary = {
        "data_label": "synthetic diagnostics only",
        "executed": requested,
        "seed": seed,
        "results": {name: str(output_root / name / "metrics.json") for name in results},
    }
    _save_json(output_root / "run_summary.json", summary)
    return summary


__all__ = ["run_experiments", "run_m0", "run_m1", "run_m2", "run_m3", "run_m4", "run_m5"]
