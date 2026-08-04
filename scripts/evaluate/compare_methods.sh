#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

conda run -n tmdpolicy tmd-policy evaluate compare --config configs/experiments/motivation.yaml "$@"

