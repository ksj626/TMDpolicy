#!/usr/bin/env python
"""Fail-closed verification of the fixed TMDpolicy runtime contract."""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from importlib.metadata import version

import torch


def main() -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.pi05 import PI05Policy
    from lerobot.policies.smolvla import SmolVLAPolicy
    from tmd_policy.backends.lerobot.compatibility import verify_installed_lerobot

    checks = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "lerobot": version("lerobot"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "hf_home": os.environ.get("HF_HOME"),
        "hf_lerobot_home": os.environ.get("HF_LEROBOT_HOME"),
        "imports": {
            "PI05Policy": PI05Policy.__module__,
            "SmolVLAPolicy": SmolVLAPolicy.__module__,
            "LeRobotDataset": LeRobotDataset.__module__,
            "libero": importlib.import_module("libero").__file__,
        },
        "compatibility": verify_installed_lerobot(),
    }
    errors = []
    if sys.version_info[:2] != (3, 12):
        errors.append("Python must be 3.12")
    if platform.machine() != "x86_64" or sys.platform != "linux":
        errors.append("native Linux x86_64 is required")
    if not checks["cuda_available"]:
        errors.append("CUDA is unavailable")
    if not checks["bf16_supported"]:
        errors.append("the CUDA device does not report BF16 support")
    if checks["lerobot"] != "0.6.1":
        errors.append("LeRobot must be 0.6.1")
    if checks["mujoco_gl"] != "egl":
        errors.append("set MUJOCO_GL=egl for headless LIBERO")
    checks["errors"] = errors
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
