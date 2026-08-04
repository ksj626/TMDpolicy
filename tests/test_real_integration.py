from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch


RUN_REAL = os.environ.get("TMD_RUN_INTEGRATION") == "1"


@pytest.mark.integration
@pytest.mark.skipif(not RUN_REAL or not torch.cuda.is_available(), reason="set TMD_RUN_INTEGRATION=1 with cached real assets and CUDA")
def test_real_pi05_fixed_noise_flow_parity(tmp_path: Path) -> None:
    from tmd_policy.integration.pi05_flow_parity import run

    root = Path(__file__).resolve().parents[1]
    manifest = root / "artifacts/data/libero_expert/episode_manifest.json"
    if not manifest.exists():
        pytest.skip("build the immutable lerobot/libero episode manifest first")
    report = run(root / "configs/teacher/pi05_flow_parity.yaml", tmp_path / "parity", sample_index=0)
    assert report["cache_unchanged"]
    assert report["deterministic_repeatability"]["maximum_absolute"] == 0.0
    assert report["normalized_32d"]["maximum_absolute"] < 1e-5
    assert report["score"]["finite_fraction"] == 1.0
