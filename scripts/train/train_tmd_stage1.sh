#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export MUJOCO_GL=egl HF_HOME="$PWD/.cache/huggingface" HF_LEROBOT_HOME="$PWD/.cache/lerobot"
conda run -n tmdpolicy python -m tmd_policy.training.preflight --config configs/methods/tmd_stage1_action_head.yaml
conda run -n tmdpolicy tmd-policy train tmd-stage1 --config configs/methods/tmd_stage1_action_head.yaml "$@"
