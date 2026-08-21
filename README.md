# TMDpolicy

TMDpolicy is a focused DMD2 robot-policy distillation codebase. It trains a
full-action-expert SmolVLA student from a frozen PI0.5 LIBERO teacher with a
PI0.5-initialized fake-score suffix, an intermediate-feature GAN, five-to-one
TTUR, denoise--renoise backward simulation, and asynchronously refreshed
student-state replay.

The supported baselines are immutable PI0.5, official 10-step SmolVLA, and
explicit 4/1-step SmolVLA ablations. Evaluation supports standard LIBERO and
the 10,030-task LIBERO-Plus benchmark. Transition Matching, Flow-SFT,
occupancy-weighted objectives, and cached/lightweight DMD2 variants are not
part of this repository.

See [DMD2 fidelity and architecture](docs/dmd2_fidelity.md) for the equations,
checkpoint contract, replay routing, precision rules, and diagnostic schema.

## Setup

All commands operate inside this repository. The wrappers use the `tmdpolicy`
Conda environment and repository-local caches.

```bash
cd /home/dmsdmswns/TMDpolicy
bash scripts/setup/create_environment.sh
bash scripts/data/build_libero_expert.sh
bash scripts/data/query_pi05_teacher.sh
bash scripts/preflight/preflight_dmd2.sh
```

LIBERO-Plus uses a separate environment so its `libero` package does not
replace standard LIBERO:

```bash
bash scripts/setup/create_libero_plus_environment.sh
```

## Train and resume

The sole training config is `configs/methods/dmd2_flow.yaml`. New runs train
the complete 99.85M-parameter SmolVLA action expert while keeping the VLM and
observation-prefix encoder frozen. Existing head-only `dmd2_final` inference
checkpoints remain evaluable, but their full optimizer checkpoints cannot be
resumed into this larger trainable-parameter contract.

```bash
bash scripts/train/train_dmd2_flow.sh \
  --output artifacts/training/dmd2_run
```

Resume only from a full checkpoint:

```bash
bash scripts/train/train_dmd2_flow.sh \
  --output artifacts/training/dmd2_run \
  --resume artifacts/training/dmd2_run/checkpoints/latest.pt
```

To bootstrap a new run from a completed balanced replay round and skip the
blocking initial collection:

```bash
bash scripts/train/train_dmd2_flow.sh \
  --initial-rollout-replay artifacts/training/dmd2_final/student_rollout_replay/round-000000 \
  --output artifacts/training/dmd2_bootstrapped
```

Training writes full checkpoints, inference-only student deltas,
`metrics.jsonl`, an atomically refreshed `training_progress.png`, validation
videos, and asynchronous replay status/logs.

## Standard LIBERO evaluation

Evaluate a trained DMD2 checkpoint on a short sample:

```bash
bash scripts/evaluate/evaluate_dmd2.sh \
  --checkpoint artifacts/training/dmd2_final/inference_checkpoints/step-00001000.pt \
  --checkpoint-sha256 auto \
  --device cuda:2 \
  --suite libero_spatial \
  --task-ids 0 1 \
  --reset-seeds 0 \
  --max-episode-steps 20 \
  --output artifacts/evaluation/dmd2_smoke
```

`--outer-steps 1` is the one-step DMD2 ablation. Repeat `--suite` to select
multiple suites. Without sampling overrides, the config evaluates all four
suites with their configured horizons.

Baselines:

```bash
bash scripts/evaluate/evaluate_pi05.sh --device cuda:2
bash scripts/evaluate/evaluate_smolvla10.sh --device cuda:2
```

The 4/1-step SmolVLA ablations use the same command directly with
`configs/evaluation/smolvla_4step_ablation.yaml` or
`configs/evaluation/smolvla_1step_ablation.yaml`.

Audit every prepared input tensor in one real episode:

```bash
bash scripts/evaluate/debug_libero_model_inputs.sh \
  --suite libero_spatial --task-id 0 --reset-seed 0 \
  --max-episode-steps 20 \
  --output artifacts/evaluation/input_audit
```

## LIBERO-Plus evaluation

Full DMD2 evaluation can shard independent serial task loops across GPUs and
resume completed tasks:

```bash
bash scripts/evaluate/evaluate_libero_plus_dmd2.sh \
  --checkpoint artifacts/training/dmd2_final/inference_checkpoints/step-00001000.pt \
  --checkpoint-sha256 auto \
  --devices cuda:2 cuda:3 cuda:4 cuda:5 \
  --output artifacts/evaluation/libero_plus_dmd2 \
  --resume
```

Use `--suite libero_spatial` for one full suite or
`--sample-per-category 10 --sample-seed 0` for a 70-task balanced performance
check (videos default on for sampled runs).

Run LIBERO-Plus baselines with:

```bash
bash scripts/evaluate/evaluate_libero_plus_baselines.sh smolvla10 --devices cuda:2 cuda:3
bash scripts/evaluate/evaluate_libero_plus_baselines.sh smolvla4 --devices cuda:2 cuda:3
bash scripts/evaluate/evaluate_libero_plus_baselines.sh smolvla1 --devices cuda:2 cuda:3
bash scripts/evaluate/evaluate_libero_plus_baselines.sh pi05 --devices cuda:2 cuda:3
```

## Tests

The default suite is CPU-only; the real model/environment integration test is
opt-in and performs no training.

```bash
conda run --no-capture-output -n tmdpolicy python -m pytest -q
TMD_RUN_INTEGRATION=1 conda run --no-capture-output -n tmdpolicy \
  python -m pytest -q tests/test_real_integration.py
```

## Public CLI

The intentional production surface is:

```text
data build-expert
teacher validate-pi05-flow
train dmd2-flow
rollout collect-student
evaluate libero
evaluate libero-plus
evaluate debug-libero-inputs
```
