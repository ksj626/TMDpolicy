#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec conda run -n lerobot env PYTHONPATH="$PROJECT_ROOT/src" python -m tmd_policy.research_cli train-tmd-plain-gaussian-ablation --config "$PROJECT_ROOT/configs/experiments/tmd_plain_gaussian_ablation.yaml" "$@"
