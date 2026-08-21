#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
conda run --no-capture-output -n tmdpolicy python -m tmd_policy.training.preflight --config configs/methods/dmd2_flow.yaml
