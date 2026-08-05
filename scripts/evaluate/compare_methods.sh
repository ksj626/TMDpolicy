#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

conda run --no-capture-output -n tmdpolicy tmd-policy evaluate compare --config configs/experiments/motivation.yaml "$@"

