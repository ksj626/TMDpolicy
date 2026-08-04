#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
conda run -n tmdpolicy tmd-policy evaluate libero --config configs/evaluation/pi05_official.yaml "$@"
