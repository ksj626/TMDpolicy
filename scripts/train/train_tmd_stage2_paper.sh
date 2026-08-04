#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
bash scripts/preflight/preflight_tmd.sh
conda run -n tmdpolicy tmd-policy train tmd-stage2 --config configs/methods/tmd_stage2_paper.yaml "$@"
