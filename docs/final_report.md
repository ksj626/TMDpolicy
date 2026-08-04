# Implementation report (2026-08-05)

## Delivered scope

The changed code is concentrated in `src/tmd_policy/methods`, the PI0.5 and
SmolVLA backends, training builders/engine/preflight, rollout/data/evaluation,
production configs and launchers, tests, and the affected documentation. New
modules own shared flow math, paper/cached discriminators, v2 replan storage,
balanced sampling, and preflight checks.

Discovered faults included the former Flow-SFT term in DMD2; a clean action-only
GAN; one normalization shared incorrectly between DMD2 and TMD-v; discarded
SmolVLA base velocity; causal chunk attention; Stage-2 expert-block refreezing;
lossy v1 rollouts; non-stratified occupancy loading; repeated-start-state
occupancy scoring; missing direct PI0.5 evaluation; accidental four-step
SmolVLA baseline evaluation; an undefined DMD2 builder variable; parity not
forwarding `video_backend`; inference tensors incompatible with version
counters; and cached-prefix PI0.5 suffix attention using SDPA instead of eager
attention.

## Mathematical and systems changes

- DMD2 uses a differentiable multi-step SmolVLA generator, a frozen PI0.5 real
  model, an exactly initialized trainable PI0.5 suffix, 5:1 TTUR, noised
  fake-score features, and no SFT. Its stopped direction is
  `(mu_fake-mu_real)/(mean_valid(abs(x-mu_real))+epsilon)`.
- TMD Stage 2 uses its distinct appendix direction
  `(g_fake-g_teacher)/(sum_valid(abs(g_fake-g_teacher))+epsilon)`, real-data
  outer transitions, and a differentiable inner rollout. It also has no SFT.
- TM-MF now uses `y=x1-x`, `h=b+Delta`, and `u=z-h`; shifted/discrete times,
  exact JVP, condition dropout, adaptive normalization, and bidirectional chunk
  attention are explicit. Zero `Delta` integrates to the SmolVLA base transition
  for 1, 2, and 4 inner steps.
- Conditional GAN and historical short-window occupancy-ratio objectives are
  separate programs. Occupancy units are actual replan observations plus full
  `[50,7]` plans, and source/task-balanced sampling is exactly resumable.
- Checkpoints contain complete program/optimizer/scheduler/RNG state, explicit
  sampler type/config/cursor, trainable identities, immutable revisions, and
  resolved upstream SHA-256 values.

## Validation performed

- Full lightweight suite: `39 passed, 1 skipped` (the opt-in real-integration
  marker), with no failures.
- All 24 production YAML configs loaded through strict fail-closed validation.
- Python compile, shell syntax, executable launcher modes, and `git diff
  --check` passed.
- Device preflight passed on the detected two RTX A5000 GPUs (23.67 GiB each).
- A real pinned PI0.5/LIBERO one-batch parity query passed: immutable cache,
  deterministic repeat maximum error `0`, finite score, and maximum normalized
  wrapper/reference error `0.00425` (`0.003448` after postprocessing).
- A real PI0.5 cloned-suffix/intermediate-feature backward probe passed with
  nonzero action gradient, selected layers `[5,11,17]`, no teacher gradients,
  693,422,112 trainable suffix parameters, and 3,449,982,704 shared frozen
  parameters.

No long training, LIBERO environment rollout/evaluation, full DMD2/TMD backward,
or real SmolVLA-10 plan was run, honoring the minimal-test constraint. The
opt-in test explains how to enable real cached-asset validation; these omitted
checks are not represented as successful results.

## Fidelity and resources

The remaining paper-fidelity limitation is architectural: the TMD head is a
SmolVLA action-space split-transformer adaptation, not the paper's invasive
native last-K video-DiT split. This is labeled in config/provenance. The selected
math, feature sources, gradients, and loss exclusions are implemented without a
silent lightweight fallback.

Paper DMD2/TMD Stage 2 use `cuda:0` for SmolVLA and `cuda:1` for the PI0.5
teacher/trainable suffix/features. Both must be nominal 24-GiB-class devices;
optimizer state and activations add substantially to parameter-only memory.
Preflight fails with an explicit diagnostic if this layout is unavailable.

## Commands

```bash
# Preflight
bash scripts/preflight/preflight_dmd2.sh
bash scripts/preflight/preflight_tmd.sh

# Training
bash scripts/train/train_dmd2_flow_paper.sh
bash scripts/train/train_tmd_stage1.sh
bash scripts/train/train_tmd_stage2_paper.sh
bash scripts/train/run_tmd_pipeline.sh \
  artifacts/training/tmd_stage1/checkpoints/final.pt \
  artifacts/training/tmd_stage2_pipeline

# Evaluation
bash scripts/evaluate/evaluate_pi05.sh
bash scripts/evaluate/evaluate_smolvla10.sh
bash scripts/evaluate/evaluate_dmd2.sh
bash scripts/evaluate/evaluate_tmd.sh

# All four suites / all 40 tasks
bash scripts/data/collect_all_libero_rollouts.sh
```

Faithful DMD2 and TMD Stage 2 contain only distribution matching plus weighted
non-saturating GAN generator terms—no Flow-SFT/data loss. Their default configs
select the PI0.5 fake score and intermediate-feature discriminator. DMD2 records
`fake_score_features`; TMD Stage 2 records `teacher_features`.
