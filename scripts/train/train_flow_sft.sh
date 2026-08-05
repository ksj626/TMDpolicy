#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export MUJOCO_GL=egl
export HF_HOME="$PWD/.cache/huggingface"
export HF_LEROBOT_HOME="$PWD/.cache/lerobot"
conda run --no-capture-output -n tmdpolicy tmd-policy train flow-sft --config configs/methods/flow_sft.yaml "$@"

