#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

stage1_checkpoint="${1:-}"
stage2_output="${2:-artifacts/training/tmd_stage2_pipeline}"
if [[ -e "$stage2_output" ]]; then
  echo "Refusing to overwrite Stage-2 output: $stage2_output" >&2
  exit 2
fi
if [[ -z "$stage1_checkpoint" ]]; then
  bash scripts/train/train_tmd_stage1.sh
  stage1_checkpoint="artifacts/training/tmd_stage1/checkpoints/final.pt"
fi
if [[ ! -f "$stage1_checkpoint" ]]; then
  echo "Stage-1 checkpoint does not exist: $stage1_checkpoint" >&2
  exit 2
fi

mkdir -p "$stage2_output"
resolved_config="$stage2_output/stage2_input_resolved.yaml"
conda run --no-capture-output -n tmdpolicy python - "$stage1_checkpoint" "$resolved_config" <<'PY'
import hashlib
import pathlib
import sys
import torch
import yaml

checkpoint = pathlib.Path(sys.argv[1]).resolve()
target = pathlib.Path(sys.argv[2])
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
if payload.get("format") != "tmdpolicy.training/v1" or payload["config"].get("method") != "tmd_stage1":
    raise SystemExit("Stage-1 provenance is incompatible: expected a tmd_stage1 v1 checkpoint")
template = yaml.safe_load(pathlib.Path("configs/methods/tmd_stage2_paper.yaml").read_text())
for key in ("models", "dataset"):
    if payload["config"][key] != template[key]:
        raise SystemExit(f"Stage-1 {key} provenance does not match Stage-2")
if payload["config"].get("tmd") != template["stage1_architecture"]:
    raise SystemExit("Stage-1 TMD architecture/sampling config does not match Stage-2")
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
template["stage2"]["stage1_checkpoint"] = str(checkpoint)
template["stage2"]["stage1_checkpoint_sha256"] = digest
target.write_text(yaml.safe_dump(template, sort_keys=False))
PY

bash scripts/preflight/preflight_tmd.sh
conda run -n tmdpolicy tmd-policy train tmd-stage2 --config "$resolved_config" --output "$stage2_output/run"

final_checkpoint="$stage2_output/run/checkpoints/final.pt"
evaluation_resolved="$stage2_output/evaluation_resolved.yaml"
conda run --no-capture-output -n tmdpolicy python - "$final_checkpoint" "$evaluation_resolved" <<'PY'
import hashlib
import pathlib
import sys
import torch
import yaml

checkpoint = pathlib.Path(sys.argv[1]).resolve()
target = pathlib.Path(sys.argv[2])
if not checkpoint.is_file():
    raise SystemExit(f"Stage-2 training did not produce the expected checkpoint: {checkpoint}")
payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
if payload.get("format") != "tmdpolicy.training/v1" or payload["config"].get("method") != "tmd_stage2":
    raise SystemExit("Stage-2 provenance is incompatible: expected a tmd_stage2 v1 checkpoint")
template = yaml.safe_load(pathlib.Path("configs/evaluation/tmd_stage2.yaml").read_text())
for key in ("models", "dataset"):
    if payload["config"][key] != template[key]:
        raise SystemExit(f"Stage-2 {key} provenance does not match evaluation")
template["policy"]["checkpoint"] = str(checkpoint)
template["policy"]["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
target.write_text(yaml.safe_dump(template, sort_keys=False))
PY
