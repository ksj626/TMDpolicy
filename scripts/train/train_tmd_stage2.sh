#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
exec bash scripts/train/train_tmd_stage2_paper.sh "$@"
