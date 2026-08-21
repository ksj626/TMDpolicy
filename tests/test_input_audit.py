from __future__ import annotations

import json

import pytest
import torch

from tmd_policy.cli import build_parser
from tmd_policy.evaluation.input_audit import tensor_summary


def test_tensor_summary_is_json_safe_and_reports_nonfinite_values() -> None:
    summary = tensor_summary(torch.tensor([1.0, float("nan"), float("inf")]))

    assert summary["shape"] == [3]
    assert summary["finite_fraction"] == pytest.approx(1 / 3)
    assert summary["min"] == 1.0
    assert summary["max"] == 1.0
    json.dumps(summary, allow_nan=False)


def test_debug_libero_inputs_cli_defaults() -> None:
    args = build_parser().parse_args(
        ["evaluate", "debug-libero-inputs", "--output", "artifacts/debug/audit"]
    )

    assert args.suite == "libero_spatial"
    assert args.task_id == 0
    assert args.reset_seed == 0
    assert args.max_episode_steps is None
    assert args.execution_horizon is None
    assert args.save_images is True
