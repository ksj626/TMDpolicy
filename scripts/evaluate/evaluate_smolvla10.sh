#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
conda run --no-capture-output -n tmdpolicy tmd-policy evaluate libero --config configs/evaluation/smolvla_official10.yaml "$@"
