"""Paired success comparison for motivation and main LIBERO evaluations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["suite"]), int(row["task_id"]), int(row["reset_seed"])


def _mcnemar(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_only = int(np.sum((left == 1) & (right == 0)))
    right_only = int(np.sum((left == 0) & (right == 1)))
    discordant = left_only + right_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, index) for index in range(0, min(left_only, right_only) + 1))
        p_value = min(1.0, 2.0 * tail / (2**discordant))
    return {"baseline_only": left_only, "method_only": right_only, "exact_two_sided_p": p_value}


def _table(report: dict[str, Any], name: str) -> dict[tuple[str, int, int], int]:
    table: dict[tuple[str, int, int], int] = {}
    for row in report["episodes"]:
        key = _key(row)
        if key in table:
            raise ValueError(f"{name} contains duplicate paired episode key {key}")
        table[key] = int(bool(row["success"]))
    return table


def _paired_statistics(
    keys: list[tuple[str, int, int]],
    tables: dict[str, dict[tuple[str, int, int], int]],
    *,
    baseline_name: str,
    bootstrap_resamples: int,
    confidence: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    baseline = np.asarray([tables[baseline_name][key] for key in keys], dtype=np.float64)
    comparisons = {}
    alpha = (1.0 - confidence) / 2.0
    for name, table in tables.items():
        values = np.asarray([table[key] for key in keys], dtype=np.float64)
        difference = values - baseline
        samples = rng.integers(0, len(keys), size=(bootstrap_resamples, len(keys)))
        draws = difference[samples].mean(axis=1)
        comparisons[name] = {
            "success_rate": float(values.mean()),
            "difference_from_baseline": float(difference.mean()),
            "paired_bootstrap_interval": [
                float(np.quantile(draws, alpha)),
                float(np.quantile(draws, 1.0 - alpha)),
            ],
            "mcnemar": _mcnemar(baseline, values),
        }
    return {"pair_count": len(keys), "comparisons": comparisons}


def compare(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    inputs = {
        name: json.loads(Path(path).read_text(encoding="utf-8")) for name, path in config["inputs"].items()
    }
    tables = {name: _table(report, name) for name, report in inputs.items()}
    baseline_name = "baseline"
    if baseline_name not in tables:
        raise ValueError("comparison inputs must contain a 'baseline' evaluation")
    if config["statistics"]["pairing_keys"] != ["suite", "task_id", "reset_seed"]:
        raise ValueError("the audited comparison key is exactly suite/task_id/reset_seed")
    keys = sorted(tables[baseline_name])
    for name, table in tables.items():
        if sorted(table) != keys:
            raise ValueError(f"{name} does not have the same paired suite/task/reset-seed grid")
    if not keys:
        raise ValueError("comparison inputs contain no episodes")
    stats = config["statistics"]
    rng = np.random.default_rng(int(stats["seed"]))
    arguments = {
        "tables": tables,
        "baseline_name": baseline_name,
        "bootstrap_resamples": int(stats["bootstrap_resamples"]),
        "confidence": float(stats["confidence"]),
        "rng": rng,
    }
    overall = _paired_statistics(keys, **arguments)
    suites = {
        suite: _paired_statistics([key for key in keys if key[0] == suite], **arguments)
        for suite in sorted({key[0] for key in keys})
    }
    tasks = {
        f"{suite}:{task_id}": _paired_statistics(
            [key for key in keys if key[0] == suite and key[1] == task_id], **arguments
        )
        for suite, task_id in sorted({(key[0], key[1]) for key in keys})
    }
    report = {
        "data": "paired real LIBERO complete episodes",
        "pairing_keys": config["statistics"]["pairing_keys"],
        "confidence": float(stats["confidence"]),
        "bootstrap_resamples": int(stats["bootstrap_resamples"]),
        **overall,
        "per_suite": suites,
        "per_task": tasks,
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


__all__ = ["compare"]
