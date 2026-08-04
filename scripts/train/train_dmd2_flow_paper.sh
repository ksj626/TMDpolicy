#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
bash scripts/preflight/preflight_dmd2.sh
conda run -n tmdpolicy tmd-policy train dmd2-flow --config configs/methods/dmd2_flow_paper.yaml "$@"
