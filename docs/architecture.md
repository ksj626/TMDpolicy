# Architecture and mathematical contract

Backends own official processors, model calls, immutable condition caches, and
coordinate conversion. Method programs own losses and optimizer phases. The
training engine owns deterministic sampling, accumulation, checkpoints, RNG,
and resume. No production path patches LeRobot.

The large PI0.5/SmolVLA backbones retain checkpoint-native mixed precision.
Repository-owned PI0.5 action/time projections, MeanFlow JVP heads, and
discriminator heads execute in FP32 even inside a BF16 training step, preventing
autocast from silently reducing the precision of clean predictions and GAN
action gradients.

## Coordinates and cache identity

Canonical LIBERO actions are `[B,50,7]`. SmolVLA and PI0.5 internal flow values
are float32 `[B,50,32]`. `ActionCoordinateBridge` unnormalizes the student,
renormalizes for PI0.5, pads dimensions 7–31, and preserves autograd. Scalar
losses mask terminal timesteps and dimensions 7–31.

PI0.5 prefix vision/language computation and KV state are cached once per
observation. Storage, shape, dtype, and tensor-version fingerprints detect
mutation. Intermediate suffix features are captured from explicitly configured
action-expert layer modules. Teacher parameters remain frozen, while eager
attention preserves gradients from layer features to fake actions.

The PI0.5 fake score shares that immutable prefix and deep-copies only the Gemma
action expert plus action/time projections. Its copied parameters are checked
against the teacher at initialization and are the only fake-score optimizer
parameters. SmolVLA cached features concatenate detached pooled visual,
language, and state prefix inputs and record model/processor revision, layer,
dtype, and dimension.

## Objectives

For each valid coordinate, corruption is `x_t=(1-t)x+t epsilon`. DMD2 backward
simulation starts at Gaussian noise, predicts
`x_hat=x_t-t*v_student(x_t,t)`, and independently re-noises the prediction at
the next checkpoint-shifted time. Prefix steps are stopped and only a randomly
selected denoising step is differentiated. Evaluation traverses the entire same
denoise--renoise chain. DMD2 uses
`stopgrad((mu_fake-mu_real)/(mean_valid(abs(x-mu_real))+epsilon))`. TMD-v instead uses

```text
g = stopgrad((g_fake - g_teacher) /
             (sum_valid(abs(g_fake-g_teacher)) + epsilon))
```

with a per-sample denominator. The two normalization modes are explicit and
method-locked in config. DMD/TMD generator loss is only VSD plus weighted
non-saturating GAN. Separate layer classifiers use the arithmetic mean of their
per-layer logistic losses.

TM-MF defines target transition `y=x1-x`, SmolVLA base `b`, independent inner
source `z`, residual `Delta`, transition `h=b+Delta`, and average velocity
`u=z-h`. The flow update is `y_r=y_s+(r-s)u`. Exact JVP builds the stopped
MeanFlow target. A zero residual integrates to `b` for every supported inner
step count. For non-flow-matching rows, `r` is exactly the largest shifted inner
student-grid value no greater than continuous shifted `s`. Adaptive loss is
`||e||^2_valid / stopgrad(||e||^2_valid + scale*N_valid)`; scale `1.0` gives
`350` for a full `[50,7]` chunk. This conditional adaptation has no CFG.

Fake-score-feature DMD2 guidance performs five joint updates, each minimizing
fake-score denoising plus an independently weighted classifier loss. TMD Stage 2
uses teacher features, so its disjoint fake-score and classifier optimizers keep
explicit update ratios. The engine rejects parameters owned by multiple
optimizers.

Scheduler time is optimizer-local: for a phase appearing `k` times in the
global phase schedule, total and warmup updates are `max_steps*k` and
`warmup_steps*k`. Checkpoint counters persist the actual update count of every
optimizer.

## Rollouts and resume

Rollout v2 stores only observed replan-start states/cameras, full policy plans,
and actual executed prefixes. It never synthesizes future states. Tensor payload
and JSON index replacement are atomic. Old schemas are rejected.

DMD2 owns a bounded per-task replay over those records. Its first collection is
blocking and covers all 40 tasks. Later snapshot collectors are separate
processes, launched every 500 actual generator updates; optimization polls and
atomically ingests completed rounds without waiting.

Ordinary training uses deterministic permutations. Occupancy training supplies
a deterministic source-paired/task-stratified batch sampler. Checkpoints contain
program, every optimizer/scheduler/scaler, trainable names, epoch and exact batch
cursor, Python/NumPy/Torch/CUDA RNG, resolved config, and provenance.
