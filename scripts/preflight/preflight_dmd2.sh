#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
conda run -n tmdpolicy python -m tmd_policy.training.preflight --config configs/methods/dmd2_flow_paper.yaml
