#!/usr/bin/env bash
#SBATCH --job-name=tmdpolicy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${1:?usage: sbatch launch_slurm.sh CONFIG --execute}"
shift
exec conda run -n lerobot env PYTHONPATH="$PROJECT_ROOT/src" python -m tmd_policy.research_cli slurm --config "$CONFIG" "$@"

