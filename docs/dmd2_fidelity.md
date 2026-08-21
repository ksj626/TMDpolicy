# DMD2 fidelity and architecture

## Objective and sampler

The implementation uses rectified-flow coordinates

```text
x_t = (1-t) x_0 + t epsilon
x_hat_0(x_t,t) = x_t - t v(x_t,t)
```

and the stopped DMD2 direction

```text
g = stopgrad((x_hat_fake - x_hat_teacher) /
             (mean_valid(abs(x_generated - x_hat_teacher)) + epsilon)).
```

The generator objective is `L_DMD + gan_weight * L_GAN`; there is no expert
regression or Flow-SFT term. The fake score is a trainable copy of the PI0.5
action-expert suffix initialized from the frozen teacher. The GAN has one FP32
classifier head per configured fake-score layer (5, 11, and 17), and averages
their losses/logits.

Training and inference call the same denoise--renoise transition code. From
pure Gaussian noise:

```text
x_hat_i = x_ti - t_i v_student(x_ti,t_i)
x_t(i+1) = (1-t_(i+1)) x_hat_i + t_(i+1) epsilon_i.
```

Training samples a step `j`, evaluates preceding transitions without gradient,
and differentiates the clean prediction at `j`. Inference evaluates every
transition. One outer step is therefore one clean prediction from Gaussian
noise, not Euler integration and not a noised expert action.

## Ownership and update schedule

Registered module names are checkpoint-visible and stable:

- `student`: frozen SmolVLA VLM/prefix encoder plus the trainable action-expert
  transformer and action/time projections;
- `teacher`: frozen PI0.5 (its parameters are excluded from inference deltas);
- `fake_score`: independently trainable PI0.5 suffix;
- `discriminator`: intermediate-feature heads;
- `bridge`: frozen student/teacher coordinate transforms.

The student trainable set contains every
`vlm_with_expert.lm_expert.*` parameter plus `action_in_proj`,
`action_out_proj`, `action_time_mlp_in`, and `action_time_mlp_out`: 99,849,312
parameters in the pinned checkpoint. `state_proj`, the vision-language model,
and the vision encoder remain frozen. The guidance optimizer owns the fake
suffix and discriminator. The generator optimizer owns only this complete
action-expert set. Cross-optimizer overlap is rejected.

The phase schedule is five guidance updates followed by one generator update.
Schedulers count actual optimizer updates:

```text
total_updates[name] = max_steps * count(name in phase_schedule)
warmup_updates[name] = warmup_steps * count(name in phase_schedule).
```

## Conditional state routing and replay

The initial replay collection is blocking and balanced across all 40 standard
LIBERO tasks. Periodic refreshes start after the configured number of actual
generator updates and run asynchronously from immutable inference-delta
snapshots. Training continues while workers collect; a completed round is
validated and atomically ingested into a bounded task-balanced buffer.

Round `r` uses simulator seed `base_reset_seed + r` and init-state index
`r mod 50`. Suite horizons are explicit and persisted. Validation rollouts use
held-out seed/init-state settings, are excluded from replay, and save videos.

Routing inside a guidance update is deliberate:

- fake-score matching uses student-replay observations and generated actions;
- GAN real/fake classification uses the expert minibatch and expert condition;
- generator DMD and GAN losses use replay observations once replay is ready.

Before replay is available, the expert minibatch is the fail-closed initial
condition source.

## Precision and safeguards

Flow coordinates, corruptions, clean predictions, DMD normalization,
discriminator heads, and captured action gradients remain FP32. PI0.5 and
SmolVLA keep their checkpoint-native internal precision. Wrappers disable
ambient autocast around FP32 projections and cast explicitly at BF16 expert
boundaries.

The GAN input gradient is captured at successively safer backward scales and
replayed through a first-order surrogate. Every phase checks finiteness,
nonzero gradients, frozen teacher/backbone parameters, and optimizer ownership.

## Checkpoints and exact resume

Full checkpoints use `tmdpolicy.training/v1` and retain `student.*`,
`fake_score.*`, `discriminator.*`, `bridge.*`, guidance/generator optimizer
states, scheduler states, counters, RNG state, and the resolved config.
Inference checkpoints use `tmdpolicy.inference/v1` and contain the complete
action-expert delta (153 tensors for the pinned checkpoint) plus its manifest
and immutable base-model contract. These checkpoints are consequently much
larger than the historical head-only deltas.

`artifacts/training/dmd2_final` remains inference-compatible: its method tag and
historical `head_only` delta are recognized and loaded using the contract stored
inside that checkpoint. Its full optimizer checkpoint is intentionally not
resume-compatible with new `action_expert` training because the generator
optimizer now owns 153 tensors rather than eight. Resume requires an equivalent
new-run config and full checkpoint; evaluation accepts either contract and
applies only its validated student delta.

## Diagnostics

`metrics.jsonl` records global/optimizer update counters, optimizer LR and
scheduler steps, fake-score/discriminator/DMD/GAN/total losses, gradient norms
and nonfinite fractions, safe GAN backward scale, random backward-simulation
step/time/noise/prefix/transition statistics, generated action dimensions and
smoothness, fake velocity/timestep/teacher tracking, DMD direction and
normalization quantiles, discriminator logits/probabilities/accuracy/AUC and
per-layer/time-bin results, GAN input-gradient norms, and DMD/GAN gradient
cosine similarity.

Fixed observation/noise validation probes track generator, teacher, and fake
score behavior. Training also refreshes `training_progress.png` atomically and
saves configured validation videos.

## Operations

Canonical commands and evaluation examples are in the root README. Important
rules are:

- train with `configs/methods/dmd2_flow.yaml` only;
- resume training from `checkpoints/*.pt`, never an inference delta;
- evaluate with `inference_checkpoints/*.pt` when possible;
- use the separate `tmdpolicy-libero-plus` environment for LIBERO-Plus;
- use official SmolVLA-10 for the baseline and label 4/1 steps as ablations.
