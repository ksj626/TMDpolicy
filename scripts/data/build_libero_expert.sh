#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export HF_HOME="$PWD/.cache/huggingface" HF_LEROBOT_HOME="$PWD/.cache/lerobot"
conda run --no-capture-output -n tmdpolicy tmd-policy data build-expert --config configs/data/libero.yaml "$@"

