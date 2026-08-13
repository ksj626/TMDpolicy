#!/usr/bin/env bash
set -euo pipefail

TMD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMD_PLUS_ENV_NAME="${TMD_PLUS_ENV_NAME:-tmdpolicy-libero-plus}"
cd "${TMD_PROJECT_ROOT}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HOME="${HF_HOME:-${TMD_PROJECT_ROOT}/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${TMD_PROJECT_ROOT}/.cache/lerobot}"

conda run --no-capture-output -n "${TMD_PLUS_ENV_NAME}" \
  env -u LD_LIBRARY_PATH \
  tmd-policy evaluate libero-plus \
  --config configs/evaluation/libero_plus_dmd2.yaml \
  "$@"
