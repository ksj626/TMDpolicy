#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
config="${1:-configs/evaluation/tmd_stage2.yaml}"
if [[ $# -gt 0 ]]; then shift; fi
conda run -n tmdpolicy tmd-policy evaluate libero --config "$config" "$@"
