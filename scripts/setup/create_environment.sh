#!/usr/bin/env bash
set -euo pipefail

TMD_ENV_NAME="${TMD_ENV_NAME:-tmdpolicy}"
TMD_CUDA_INDEX_URL="${TMD_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
TMD_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

conda create -n "${TMD_ENV_NAME}" python=3.12 pip -y
conda install -n "${TMD_ENV_NAME}" -c conda-forge ffmpeg -y
conda run --no-capture-output -n "${TMD_ENV_NAME}" python -m pip install \
  torch==2.11.0 torchvision==0.26.0 --index-url "${TMD_CUDA_INDEX_URL}"
conda run --no-capture-output -n "${TMD_ENV_NAME}" python -m pip install \
  "lerobot[training,pi,smolvla,libero,evaluation]==0.6.1" \
  -c "${TMD_PROJECT_ROOT}/environment/constraints.txt"
conda run --no-capture-output -n "${TMD_ENV_NAME}" python -m pip install -e "${TMD_PROJECT_ROOT}[test]"

MUJOCO_GL=egl conda run --no-capture-output -n "${TMD_ENV_NAME}" \
  python "${TMD_PROJECT_ROOT}/scripts/setup/verify_environment.py"
