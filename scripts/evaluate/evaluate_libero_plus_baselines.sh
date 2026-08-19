#!/usr/bin/env bash
set -euo pipefail

TMD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMD_PLUS_ENV_NAME="${TMD_PLUS_ENV_NAME:-tmdpolicy-libero-plus}"
cd "${TMD_PROJECT_ROOT}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export HF_HOME="${HF_HOME:-${TMD_PROJECT_ROOT}/.cache/huggingface}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${TMD_PROJECT_ROOT}/.cache/lerobot}"

usage() {
  echo "Usage: $0 {smolvla10|smolvla4|smolvla1|pi05|all} [LIBERO-Plus options]"
  echo
  echo "Examples:"
  echo "  $0 smolvla10 --devices cuda:2 cuda:3 --sample-per-category 10"
  echo "  $0 pi05 --devices cuda:4 cuda:5 --output artifacts/evaluation/pi05_plus"
  echo "  $0 all --devices cuda:2 cuda:3 --resume"
  echo
  echo "The 'all' selector uses each config's independent default output directory."
}

if [[ $# -eq 0 || "${1}" == "-h" || "${1}" == "--help" ]]; then
  usage
  exit 0
fi

selector="$1"
shift

smolvla10_config="configs/evaluation/libero_plus_smolvla_official10.yaml"
smolvla4_config="configs/evaluation/libero_plus_smolvla_4step_ablation.yaml"
smolvla1_config="configs/evaluation/libero_plus_smolvla_1step_ablation.yaml"
pi05_config="configs/evaluation/libero_plus_pi05_official.yaml"

run_config() {
  local config="$1"
  shift
  conda run --no-capture-output -n "${TMD_PLUS_ENV_NAME}" \
    env -u LD_LIBRARY_PATH \
    tmd-policy evaluate libero-plus \
    --config "${config}" \
    "$@"
}

case "${selector}" in
  smolvla10)
    run_config "${smolvla10_config}" "$@"
    ;;
  smolvla4)
    run_config "${smolvla4_config}" "$@"
    ;;
  smolvla1)
    run_config "${smolvla1_config}" "$@"
    ;;
  pi05)
    run_config "${pi05_config}" "$@"
    ;;
  all)
    for argument in "$@"; do
      if [[ "${argument}" == "--output" || "${argument}" == --output=* ]]; then
        echo "error: 'all' cannot share --output; use the four independent default output directories" >&2
        exit 2
      fi
    done
    run_config "${smolvla10_config}" "$@"
    run_config "${smolvla4_config}" "$@"
    run_config "${smolvla1_config}" "$@"
    run_config "${pi05_config}" "$@"
    ;;
  *)
    echo "error: unknown baseline '${selector}'" >&2
    usage >&2
    exit 2
    ;;
esac
