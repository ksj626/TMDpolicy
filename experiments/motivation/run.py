from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.motivation.runner import run_experiments

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pre-distillation motivation diagnostics")
    parser.add_argument("--experiments", nargs="+", default=["M0"])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "motivation")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    summary = run_experiments(args.output, args.experiments, seed=args.seed)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
