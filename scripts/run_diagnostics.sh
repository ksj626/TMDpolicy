#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/dmsdmswns/TMDpolicy
PY=/home/dmsdmswns/anaconda3/envs/lerobot/bin/python
export PYTHONPATH="$PROJECT/src"
export HF_HOME="$PROJECT/.cache/huggingface"
export HF_LEROBOT_HOME="$PROJECT/.cache/lerobot"
export MUJOCO_GL=egl

cd "$PROJECT"
"$PY" -m tmd_policy.cli inspect --config configs/tiny.yaml
"$PY" -m pytest -q
"$PY" -m tmd_policy.cli synthetic-smoke --output artifacts/smoke
"$PY" -m tmd_policy.cli build-expert --config configs/tiny.yaml --output artifacts/expert_real --max-chunks 2
