# TMDpolicy

Repository-owned PI0.5-to-SmolVLA action-policy distillation on the immutable
`lerobot/libero` dataset. Production baselines use LeRobot exactly `0.6.1` and
never modify installed LeRobot or site-packages.

Pinned assets:

- PI0.5 teacher: `lerobot/pi05_libero_finetuned@8e174154ef5f6c60a8da12ae99c303d8963138c1`
- SmolVLA student: `lerobot/smolvla_libero@31d453f7edd78c839a8bbc39744a292686daf0de`
- LIBERO data: `lerobot/libero@a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`

## Implemented baselines

`dmd2_flow_paper` uses a frozen PI0.5 teacher, a PI0.5-initialized independently
trainable action-expert suffix, five fake-score updates per generator update,
VSD, and a noised-action GAN on selected fake-score layers. Its generator loss
is exactly `L_VSD + lambda_GAN L_GAN`; it has no Flow-SFT or data regression.
The direct baseline is DMD2-v: training predicts a clean action from one
real-data outer transition, while evaluation deterministically integrates the
checkpoint's shifted descending student grid. The five guidance updates jointly
train fake-score and classifier parameters from one detached student sample.

TMD uses the repository-owned SmolVLA action-space transformer adaptation:

1. Stage 1 trains TM-MF with independent outer/inner Gaussian sources,
   a shifted discrete outer grid, continuous shifted MeanFlow `s`, the largest
   inner-grid predecessor `r<=s`, `r=s` on 75% of
   rows, exact JVP, and adaptive normalization.
2. Stage 2 loads Stage 1 immutably, preserves the selected final SmolVLA expert
   blocks, constructs real-data outer transitions, differentiates through every
   inner step, and applies TMD v2 DMD2-v VSD plus teacher-feature GAN. It has no
   SFT/data term.

For the action-space adaptation, `b=v_SmolVLA(x_t,t,c)`, the head predicts a
zero-initialized residual `Delta`, `h=b+Delta`, and the MeanFlow average velocity
is `u=z-h`. Therefore a zero residual returns `b` after 1, 2, or 4 inner steps
and reproduces the corresponding SmolVLA outer Euler transition. Chunk
attention is bidirectional by default; causal attention is an explicit ablation.

The ordinary DMD/TMD conditional GAN and the short-window occupancy ratio are
different programs. Occupancy training uses actual replan records
`(s_t,o_t,a_plan[50],task)` from expert data and real student rollouts. Both
families corrupt actions as

```text
a_tau = (1-tau) a + tau epsilon
tau = gamma u / ((gamma-1)u + 1)
```

and use logistic discriminator and non-saturating generator losses. The default
paper feature path uses separate heads on PI0.5 layers `[5,11,17]` and averages
their losses. `cached_vla_features` is a separately named efficient adaptation.

## Environment

```bash
cd /home/dmsdmswns/TMDpolicy
bash scripts/setup/create_environment.sh
conda activate tmdpolicy
export MUJOCO_GL=egl
export HF_HOME="$PWD/.cache/huggingface"
export HF_LEROBOT_HOME="$PWD/.cache/lerobot"
```

Accept access to `google/paligemma-3b-pt-224`. Every model, processor, dataset,
checkpoint, and LeRobot source identity is recorded. Checkpoint SHA-256 fields
set to `auto` are computed before use and persisted into the resolved run config.

## Data and parity

```bash
bash scripts/data/build_libero_expert.sh
bash scripts/data/query_pi05_teacher.sh --output artifacts/pi05_flow_parity
```

Rollout schema `tmdpolicy.libero-replans/v2` losslessly stores each canonical
replan-start camera tensor and metadata, state `[8]`, full postprocessed plan
`[50,7]`, executed prefix/actions, suite-local and global task identities,
termination outcome, behavior checkpoint/round, and immutable revisions. Old v1
stores fail closed. All-40-task collection is:

```bash
bash scripts/data/collect_all_libero_rollouts.sh
```

## Training

Preflight never changes the requested algorithm; insufficient memory/device
fails explicitly.

```bash
bash scripts/preflight/preflight_dmd2.sh
bash scripts/train/train_dmd2_flow_paper.sh

bash scripts/preflight/preflight_tmd.sh
bash scripts/train/train_tmd_stage1.sh
bash scripts/train/train_tmd_stage2_paper.sh
```

The pipeline either accepts an existing Stage-1 checkpoint or trains Stage 1,
computes its SHA-256, checks its model/dataset provenance, writes a fully
resolved Stage-2 input config, and refuses existing output directories. After
training it writes `evaluation_resolved.yaml` with the exact final checkpoint
path and SHA-256:

```bash
bash scripts/train/run_tmd_pipeline.sh \
  artifacts/training/tmd_stage1/checkpoints/final.pt \
  artifacts/training/tmd_stage2_pipeline
```

Paper occupancy and cached-VLA occupancy launch with:

```bash
conda run -n tmdpolicy tmd-policy train occupancy-discriminator \
  --config configs/methods/occupancy_discriminator_paper.yaml
conda run -n tmdpolicy tmd-policy train occupancy-discriminator \
  --config configs/methods/occupancy_discriminator_cached_vla.yaml
```

Training batches for occupancy are deterministic source-paired/task-stratified
and resume from an exact epoch/batch cursor. Validation is never oversampled and
reports source, macro/task, and aggregate metrics. Every trainer displays global
step and validation progress and atomically refreshes `training_progress.png` in
its output directory after each completed step.

The paper DMD2 configuration keeps effective batch size 32 as `8 × 4`, reuses
each immutable PI0.5 condition cache within a loss, and writes a lightweight
student-delta checkpoint every 50 steps under `inference_checkpoints/`.

## Evaluation

Direct PI0.5 and the ordinary SmolVLA baseline use official 10-step sampling.
SmolVLA four-step sampling is only available as an explicitly labeled ablation.

```bash
bash scripts/evaluate/evaluate_pi05.sh
bash scripts/evaluate/evaluate_smolvla10.sh
bash scripts/evaluate/evaluate_dmd2.sh
bash scripts/evaluate/evaluate_tmd.sh
```

Evaluate two spatial tasks from an intermediate DMD2 inference checkpoint:

```bash
bash scripts/evaluate/evaluate_dmd2.sh \
  --checkpoint artifacts/training/dmd2_flow_paper/inference_checkpoints/step-00000050.pt \
  --checkpoint-sha256 auto \
  --device cuda:2 \
  --suite libero_spatial \
  --task-ids 0 5 \
  --reset-seeds 0 \
  --max-episode-steps 600 \
  --output artifacts/evaluation/dmd2_step50_spatial_0_5
```

The `inference_checkpoints` files contain only the trained student delta and
are for evaluation. Use full files under `checkpoints/` with `--resume` when
continuing training. A shortened episode horizon is a smoke test, not a
reportable LIBERO success-rate evaluation.

Use `configs/evaluation/tmd_stage1.yaml` as the first argument to
`evaluate_tmd.sh` for Stage 1. Evaluation actions are canonical `[B,50,7]` and
comparison requires identical `(suite, task_id, reset_seed)` grids. Motivation
and main comparison configs include PI0.5 official, SmolVLA official-10, and the
explicit four-step ablation. Evaluation and rollout collection display both
overall episode progress, per-plan NFE progress, and the environment-step progress
of the active episode.

## Memory and fidelity

The tested PI0.5 suffix clone has 693,422,112 trainable parameters and shares
3,449,982,704 frozen teacher parameters. The default layout uses `cuda:0` for
SmolVLA and `cuda:1` for PI0.5 prefix/suffix/features; both require a nominal
24-GB-class GPU. Optimizer state and activations dominate beyond parameter-only
memory. Config preflight records detected devices and fails rather than selecting
a lightweight score or action-only discriminator.

The TMD action head is a truthful SmolVLA action-space architectural adaptation,
not the paper's native last-K video-DiT split. DMD2's teacher-residual weighting
and TMD-v's fake–teacher stopped-L1 direction are distinct, method-locked
objectives. The preconditioning, time sampling, inner rollout, feature sources,
and loss exclusions are preserved within the labeled adaptation.
The current conditional action-policy TMD has no CFG; nonzero condition dropout
is rejected rather than partially dropping only captured features.

See [architecture](docs/architecture.md), [experiment protocol](docs/experiment_protocol.md),
[config reference](configs/README.md), and the method/data/evaluation READMEs.
