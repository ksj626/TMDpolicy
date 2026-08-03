from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def _save(figure: Any, output: Path, stem: str, footer: str) -> list[str]:
    figure.text(0.01, 0.005, footer, fontsize=7, color="0.3")
    figure.tight_layout(rect=(0, 0.035, 1, 0.95))
    paths = []
    for suffix in ("png", "svg"):
        path = output / f"{stem}.{suffix}"
        figure.savefig(path, dpi=180 if suffix == "png" else None)
        paths.append(str(path))
    plt.close(figure)
    return paths


def plot_m0(data: dict[str, Any], output: Path, metadata: str) -> list[str]:
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    figure.suptitle("M0 — data-source sanity [SYNTHETIC]")
    comparisons = data["comparisons"]
    names = list(comparisons)
    aucs = [comparisons[name]["metrics"]["roc_auc"] for name in names]
    axes[0, 0].bar(names, aucs, color=["#777777", "#999999", "#3264a8"])
    axes[0, 0].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[0, 0].set_ylabel("ROC-AUC")
    axes[0, 0].tick_params(axis="x", rotation=20)
    for name in names:
        curve = comparisons[name]["curve"]
        axes[0, 1].plot(curve["fpr"], curve["tpr"], label=name)
        axes[0, 2].plot(curve["recall"], curve["precision"], label=name)
    axes[0, 1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0, 1].set(xlabel="false-positive rate", ylabel="true-positive rate", title="ROC")
    axes[0, 2].set(xlabel="recall", ylabel="precision", title="precision-recall")
    main = comparisons["expert_vs_current"]
    axes[1, 0].hist(main["positive_logits"], bins=30, alpha=0.6, label="expert")
    axes[1, 0].hist(main["negative_logits"], bins=30, alpha=0.6, label="current")
    axes[1, 0].set(xlabel="final prefix logit", ylabel="episodes", title="held-out logits")
    calibration = main["calibration"]
    axes[1, 1].plot(calibration["confidence"], calibration["accuracy"], "o-")
    axes[1, 1].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[1, 1].set(xlabel="mean confidence", ylabel="expert frequency", title="reliability")
    task_auc = main["per_task_auc"]
    axes[1, 2].bar(list(task_auc), list(task_auc.values()))
    axes[1, 2].axhline(0.5, color="black", linestyle="--", linewidth=1)
    axes[1, 2].set(xlabel="task", ylabel="ROC-AUC", title="per-task held-out AUC")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 1].legend(fontsize=7)
    axes[0, 2].legend(fontsize=7)
    axes[1, 0].legend(fontsize=8)
    return _save(figure, output, "m0_source_sanity", metadata)


def plot_m1(data: dict[str, Any], output: Path, metadata: str) -> list[str]:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    figure.suptitle("M1 — expert vs baseline occupancy [SYNTHETIC]")
    variants = data["variants"]
    names = list(variants)
    axes[0, 0].bar(names, [variants[name]["metrics"]["roc_auc"] for name in names])
    axes[0, 0].set(ylabel="held-out ROC-AUC", title="discriminator variants")
    prefix = data["prefix"]
    axes[0, 1].plot(prefix["expert_probability_mean"], label="expert")
    axes[0, 1].plot(prefix["current_probability_mean"], label="current")
    axes[0, 1].set(xlabel="prefix position", ylabel="expert probability", title="prefix curves")
    axes[1, 0].hist(prefix["expert_final_logits"], bins=30, alpha=0.6, label="expert")
    axes[1, 0].hist(prefix["current_final_logits"], bins=30, alpha=0.6, label="current")
    axes[1, 0].set(xlabel="final prefix logit", ylabel="episodes", title="final logits")
    task_auc = variants["prefix"]["per_task_auc"]
    axes[1, 1].bar(list(task_auc), list(task_auc.values()))
    axes[1, 1].set(xlabel="task", ylabel="ROC-AUC", title="prefix per-task AUC")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    axes[0, 1].legend()
    axes[1, 0].legend()
    return _save(figure, output, "m1_occupancy", metadata)


def plot_m2(data: dict[str, Any], output: Path, metadata: str) -> list[str]:
    success = np.asarray(data["success"], dtype=bool)
    logits = np.asarray(data["final_logits"])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    figure.suptitle("M2 — score and behavior quality [SYNTHETIC]")
    axes[0].boxplot([logits[~success], logits[success]], tick_labels=["failure", "success"])
    axes[0].set(ylabel="final prefix logit", title="held-out current episodes")
    axes[1].scatter(np.arange(len(logits)), logits, c=success, cmap="coolwarm", s=12)
    axes[1].set(xlabel="episode index", ylabel="final prefix logit", title="episode scores")
    for axis in axes:
        axis.grid(alpha=0.2)
    return _save(figure, output, "m2_success_relationship", metadata)


def plot_m3(data: dict[str, Any], output: Path, metadata: str) -> list[str]:
    logits = np.asarray(data["prefix_logits"])
    increments = np.asarray(data["increments"])
    success = np.asarray(data["success"], dtype=bool)
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    figure.suptitle("M3 — temporal failure localization [SYNTHETIC]")
    axes[0, 0].imshow(logits[:40], aspect="auto", cmap="coolwarm")
    axes[0, 0].set(xlabel="prefix position", ylabel="episode", title="prefix logits")
    axes[0, 1].imshow(-increments[:40], aspect="auto", cmap="magma")
    axes[0, 1].set(xlabel="transition", ylabel="episode", title="incremental mismatch -r_j")
    axes[1, 0].plot(logits[success].mean(0), label="success")
    axes[1, 0].plot(logits[~success].mean(0), label="failure")
    axes[1, 0].set(xlabel="prefix position", ylabel="logit", title="group means")
    axes[1, 1].plot(np.asarray(data["example_action_norm"]), label="action norm")
    axes[1, 1].plot(-increments[0], label="mismatch increment")
    axes[1, 1].set(xlabel="transition", ylabel="value", title="qualitative trace")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        if axis in (axes[1, 0], axes[1, 1]):
            axis.legend()
    return _save(figure, output, "m3_temporal_localization", metadata)


def plot_m4(data: dict[str, Any], output: Path, metadata: str) -> list[str]:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    figure.suptitle("M4 — perturbation proxy [SYNTHETIC; NOT A ROBOT RESULT]")
    axes[0].boxplot(
        [data["standard_final_logits"], data["perturbed_final_logits"]],
        tick_labels=["standard", "perturbed"],
    )
    axes[0].set(ylabel="final prefix logit", title="score shift")
    axes[1].bar(["standard", "perturbed"], [data["standard_success"], data["perturbed_success"]])
    axes[1].set(ylabel="synthetic success rate", title="behavior degradation")
    return _save(figure, output, "m4_synthetic_perturbation", metadata)


def plot_m5(data: dict[str, Any], output: Path, metadata: str) -> list[str]:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    figure.suptitle("M5 — coarse sampling proxy [SYNTHETIC; latency unavailable]")
    axes[0].bar(["10-step proxy", "2-step proxy"], data["success_rates"])
    axes[0].set(ylabel="synthetic success rate", title="quality")
    axes[1].bar(["difference", "diversity"], [data["action_mae"], data["coarse_diversity"]])
    axes[1].set(ylabel="canonical action units", title="action diagnostics")
    axes[2].boxplot(
        [data["original_final_logits"], data["coarse_final_logits"]],
        tick_labels=["10-step", "2-step"],
    )
    axes[2].set(ylabel="final prefix logit", title="occupancy score")
    return _save(figure, output, "m5_coarse_sampling", metadata)


__all__ = ["plot_m0", "plot_m1", "plot_m2", "plot_m3", "plot_m4", "plot_m5"]
