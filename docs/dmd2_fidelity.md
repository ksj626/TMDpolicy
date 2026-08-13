# DMD2 fidelity, diagnostics, and operations

## Theory implemented here

DMD2 distills a diffusion/flow teacher by differentiating a distribution
matching direction. With rectified-flow convention

```text
x_t = (1-t) x_0 + t epsilon
x_hat_0(x_t,t) = x_t - t v(x_t,t)
```

the frozen teacher and online fake model give clean predictions
`mu_real` and `mu_fake`. The student direction is the stopped fake-minus-real
difference, weighted by DMD2's teacher-residual denominator:

```text
g_DMD = stopgrad((mu_fake - mu_real) /
                 (mean_valid(abs(x_generated - mu_real)) + epsilon))
```

The fake model is trained on forward-noised generated samples. DMD2 removes the
original regression term and adds a non-saturating GAN. Real expert and generated
action chunks are independently noised before the discriminator. The resulting
student objective is exactly

```text
L_generator = L_distribution_matching + gan_weight * softplus(-D(fake_t))
```

with no Flow-SFT/data regression.

For multi-step generation, DMD2 does not Euler-integrate the velocity. It starts
from Gaussian noise and alternates clean prediction with independent forward
re-noising:

```text
x_t0 ~ Normal(0,I)
x_hat_i = x_ti - t_i v_student(x_ti,t_i)
x_t(i+1) = (1-t_(i+1)) x_hat_i + t_(i+1) epsilon_i
```

Training samples one step index `j`, simulates its prefix without gradients,
and differentiates only `x_hat_j`. The same shifted time grid and stochastic
denoise--renoise transitions are used during DMD2 evaluation. `outer_steps=1`
therefore means one clean prediction from pure noise.

The five-to-one update ratio is TTUR. Each optimizer's schedule is measured in
its actual updates:

```text
total_updates[name]  = max_steps * count(name in phase_schedule)
warmup_updates[name] = warmup_steps * count(name in phase_schedule)
```

For the paper config this is 500,000 guidance updates versus 100,000 generator
updates, and 5,000 versus 1,000 warmup updates.

## Robotics state-distribution adaptation

Periodic student-state replay is a repository adaptation for conditional robot
policies, not a claim about the image-generation DMD2 paper. Before the first
optimizer step, a lightweight student snapshot runs one rollout in every one of
the 40 original LIBERO tasks. This initial collection is blocking, shards the
tasks over the configured rollout GPUs, and streams progress to the terminal.

At the configured actual-generator-update interval, training writes another
immutable student snapshot and starts a coordinator whose serial workers use the
configured rollout devices. The trainer continues optimizing while they run and
ingests a round only after its merged atomic rollout store is complete. The bounded replay samples tasks in
round-robin order. Within a combined guidance update, fake-score score matching
uses replay observations and an independently sampled student action, while GAN
real/fake classification uses the expert minibatch and an expert-conditioned
student action. The generator's DMD and GAN objectives use replay observations
when the buffer is available. Before the initial replay is available, guidance
falls back to the expert batch for both branches.

Collector state is under
`<training-output>/student_rollout_replay/`: resolved round configs, asynchronous
logs, snapshots, completed rollout stores, and `status.json`. A failed periodic
refresh is reported but does not destroy the last valid replay; a failed initial
balanced collection aborts training.

## Precision and gradient contract

Public flow coordinates, corruptions, clean predictions, DMD normalization, GAN
head logits, and captured action gradients are FP32. SmolVLA and PI0.5 retain
their checkpoint-native internal mixed layout. Repository wrappers disable an
ambient autocast around FP32 action/time projections and explicitly cast at the
BF16 expert boundary. GAN action gradients are captured with successively safer
loss scales and replayed as a first-order surrogate, avoiding a second traversal
of the large suffix.

The constructor and every phase validate that the PI0.5 teacher is completely
frozen, frozen SmolVLA parameters receive no gradients, fake-score/classifier
parameters update only in their phases, and student head parameters receive a
finite nonzero generator gradient.

## Training and monitoring

The default paper run needs the expert manifest, GPUs `cuda:0` and `cuda:1` for
training, and `cuda:2` for asynchronous rollout inference:

```bash
conda activate tmdpolicy
cd /home/dmsdmswns/TMDpolicy
bash scripts/data/build_libero_expert.sh
bash scripts/preflight/preflight_dmd2.sh
bash scripts/train/train_dmd2_flow_paper.sh \
  --output artifacts/training/dmd2_flow_paper_run1
```

To start a new run from an already completed, balanced round-1 replay and skip
the blocking pre-training rollout, pass the round directory (not its parent):

```bash
bash scripts/train/train_dmd2_flow_paper.sh \
  --initial-rollout-replay /home/dmsdmswns/TMDpolicy/artifacts/training/dmd2_flow_paper_run1/student_rollout_replay/round-000000 \
  --output artifacts/training/dmd2_flow_paper_run2
```

The round must contain `collection_report.json` and train replans for all 40
original LIBERO tasks. It is loaded into the new run's bounded replay without
copying the source directory. The first asynchronous refresh remains due at 500
actual generator updates.

The output contains:

- `metrics.jsonl`: one complete row per global step;
- `training_progress.png`: atomically refreshed loss plot;
- `inference_checkpoints/step-*.pt`: student deltas for evaluation;
- `checkpoints/step-*.pt`: full optimizer/RNG state for training resume;
- `student_rollout_replay/status.json`: live collector/buffer state.

Metric prefixes include optimizer update counts/LRs/scheduler steps;
`backward/*` source/prefix/transition/time diagnostics; `fake_score/*` score
tracking and timestep bins; generator DMD direction, denominator, smoothness and
per-action-dimension statistics; `classifier/*` logits/probabilities/AUC/layer
and timestep metrics; phase and parameter-group gradient norms/nonfinite
fractions; GAN/DMD input-gradient alignment; and `validation/fixed_probe/*` for
fixed observation/noise generator, teacher, and fake-score probes.

Continue training only from a full checkpoint and use the same resolved config:

```bash
bash scripts/train/train_dmd2_flow_paper.sh \
  --output artifacts/training/dmd2_flow_paper_run1 \
  --resume artifacts/training/dmd2_flow_paper_run1/checkpoints/latest.pt
```

## Original LIBERO evaluation

Evaluate a small task sample from an intermediate student-delta checkpoint:

```bash
bash scripts/evaluate/evaluate_dmd2.sh \
  --checkpoint artifacts/training/dmd2_flow_paper_run1/inference_checkpoints/step-00000500.pt \
  --checkpoint-sha256 auto \
  --device cuda:2 \
  --suite libero_spatial \
  --task-ids 0 5 \
  --reset-seeds 0 \
  --max-episode-steps 600 \
  --output artifacts/evaluation/dmd2_step500_spatial_sample
```

Add `--outer-steps 1` for one-step DMD2 inference. Repeat `--suite` to select
multiple original suites; omitting suite/task/seed flags uses the full grid in
the evaluation config.

## LIBERO-Plus evaluation

LIBERO-Plus must not replace vanilla LIBERO in the training environment. Create
the separate pinned environment once:

```bash
bash scripts/setup/create_libero_plus_environment.sh
```

The setup defaults to fork commit
`4976dc30028e805ff8094b55501d532c48fec182`; override
`TMD_LIBERO_PLUS_COMMIT` only when intentionally starting a new benchmark
provenance. The evaluator records that commit and the classification-file hash.

Run the full four-suite, one-trial-per-variant 10,030-task benchmark. For
throughput, assign independent serial task shards to multiple GPUs:

```bash
bash scripts/evaluate/evaluate_libero_plus_dmd2.sh \
  --checkpoint artifacts/training/dmd2_flow_paper_run1/inference_checkpoints/final.pt \
  --checkpoint-sha256 auto \
  --devices cuda:2 cuda:3 cuda:4 cuda:5 \
  --output artifacts/evaluation/libero_plus_dmd2_run1
```

The coordinator launches one policy replica per device and distributes complete
tasks round-robin. Every worker uses the original batch-1 reset, plan, step, and
termination loop, so there is no cross-task synchronization or waiting for a
batch's slowest episode. Worker results are checked for exact task coverage and
merged into `episodes.jsonl`. Resume without repeating completed tasks:

```bash
bash scripts/evaluate/evaluate_libero_plus_dmd2.sh \
  --checkpoint artifacts/training/dmd2_flow_paper_run1/inference_checkpoints/final.pt \
  --checkpoint-sha256 auto \
  --devices cuda:2 cuda:3 cuda:4 cuda:5 \
  --output artifacts/evaluation/libero_plus_dmd2_run1 \
  --resume
```

Fully evaluate one selected suite, or a selected subset of the four suites:

```bash
# One complete suite.
bash scripts/evaluate/evaluate_libero_plus_dmd2.sh \
  --checkpoint artifacts/training/dmd2_flow_paper_run1/inference_checkpoints/final.pt \
  --checkpoint-sha256 auto \
  --suite libero_spatial \
  --output artifacts/evaluation/libero_plus_spatial

# Two complete suites. Repeating --suite also works.
bash scripts/evaluate/evaluate_libero_plus_dmd2.sh \
  --checkpoint artifacts/training/dmd2_flow_paper_run1/inference_checkpoints/final.pt \
  --checkpoint-sha256 auto \
  --suite libero_object libero_goal \
  --output artifacts/evaluation/libero_plus_object_goal
```

A lightweight environment smoke test (not a reportable benchmark) is:

```bash
bash scripts/evaluate/evaluate_libero_plus_dmd2.sh \
  --checkpoint artifacts/training/dmd2_flow_paper_run1/inference_checkpoints/step-00000500.pt \
  --checkpoint-sha256 auto \
  --suite libero_spatial \
  --task-ids 0 1 \
  --max-episode-steps 20 \
  --output artifacts/evaluation/libero_plus_smoke
```

For a more representative performance check, deterministically sample exactly
10 variants from each of the seven perturbation categories (70 episodes total),
balanced across the four suites:

```bash
bash scripts/evaluate/evaluate_libero_plus_dmd2.sh \
  --checkpoint artifacts/training/dmd2_flow_paper_run1/inference_checkpoints/step-00000060.pt \
  --checkpoint-sha256 auto \
  --device cuda:3 \
  --sample-per-category 10 \
  --sample-seed 0 \
  --output artifacts/evaluation/libero_plus_dmd2_category10
```

The exact suite-local task IDs are recorded under `selection.tasks` in
`manifest.json`. Category-sampled runs save agent-view videos by default under
`videos/<suite>/`; use `--no-save-videos` to disable them. Conversely, use
`--save-videos` to opt into videos for a full run. Use the same seed and
`--resume` to continue an interrupted sample without changing its task set.

The configured evaluation batch size is `1`. `--devices` is the acceleration
control; omit it (or pass one device with `--device`) for a single serial worker.
Each extra worker needs enough VRAM for one policy replica. Model startup is paid
once per worker, so multi-GPU sharding helps sustained sampled/full evaluations,
not tiny one-step smoke tests. This container exposes one EGL render device, so
rendering remains there while policy replicas use the requested CUDA devices.

## Batched DMD2 student rollout refresh

Initial and asynchronous DMD2 student-state collection use the same process-level
task sharding. Configure the worker GPUs in the training YAML:

```yaml
dmd2:
  student_rollout_replay:
    devices: [cuda:2, cuda:3, cuda:4, cuda:5]
    batch_size: 1
```

The coordinator assigns the 40 tasks across the listed devices, validates every
worker store, and merges them into the same atomic round consumed by training.
This does not change task balance, reset seeds, rollout horizons, RNG rules,
replay schema, asynchronous refresh schedule, or expert-state GAN minibatches.

Primary references are the [DMD2 paper](https://arxiv.org/abs/2405.14867),
[official DMD2 implementation](https://github.com/tianweiy/DMD2),
[LeRobot LIBERO-Plus guide](https://github.com/huggingface/lerobot/blob/main/docs/source/libero_plus.mdx),
and [LIBERO-Plus benchmark](https://github.com/sylvestf/LIBERO-plus).
