"""Fail-fast device/memory checks for paper-faithful production configurations."""

from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from tmd_policy.config import load_config


def preflight(config: dict[str, Any]) -> dict[str, Any]:
    requirements = config.get("preflight", {}).get("minimum_total_memory_gib", {})
    report: dict[str, Any] = {"requirements": requirements, "devices": {}}
    if not requirements:
        return report
    if not torch.cuda.is_available():
        raise RuntimeError(
            "paper-faithful training requires CUDA devices, but torch.cuda.is_available() is false"
        )
    for locator, required in requirements.items():
        device = torch.device(locator)
        if device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count():
            raise RuntimeError(f"configured paper component device is unavailable: {locator}")
        properties = torch.cuda.get_device_properties(device)
        total_gib = properties.total_memory / 2**30
        report["devices"][locator] = {
            "name": properties.name,
            "total_memory_gib": total_gib,
            "required_memory_gib": float(required),
        }
        if total_gib < float(required):
            raise RuntimeError(
                f"paper-faithful configuration requires at least {required} GiB on {locator} "
                f"for the selected algorithm, but {properties.name} has {total_gib:.2f} GiB; "
                "no lightweight fallback will be selected"
            )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(preflight(load_config(args.config)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["preflight"]
