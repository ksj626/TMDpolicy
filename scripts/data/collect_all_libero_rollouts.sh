#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
conda run -n tmdpolicy tmd-policy rollout collect-student --config configs/rollout/student.yaml "$@"
