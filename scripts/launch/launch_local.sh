#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: launch_local.sh CONFIG [extra arguments]}"
shift
exec conda run -n lerobot env PYTHONPATH="$PROJECT_ROOT/src" python -m tmd_policy.research_cli launch --config "$CONFIG" "$@"

