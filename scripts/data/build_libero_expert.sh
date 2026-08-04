#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec conda run -n lerobot env PYTHONPATH="$PROJECT_ROOT/src" python -m tmd_policy.research_cli build-libero-expert --config "$PROJECT_ROOT/configs/data/libero.yaml" "$@"

