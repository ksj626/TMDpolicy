# Architecture and mathematical contract

Backends own official processors, model calls, immutable condition caches, and
coordinate conversion. Method programs own losses and optimizer phases. The
training engine owns deterministic sampling, accumulation, checkpoints, RNG,
and resume. No production path patches LeRobot.

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

For each valid coordinate, corruption is `x_t=(1-t)x+t epsilon`. DMD2 uses
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
step count.

## Rollouts and resume

Rollout v2 stores only observed replan-start states/cameras, full policy plans,
and actual executed prefixes. It never synthesizes future states. Tensor payload
and JSON index replacement are atomic. Old schemas are rejected.

Ordinary training uses deterministic permutations. Occupancy training supplies
a deterministic source-paired/task-stratified batch sampler. Checkpoints contain
program, every optimizer/scheduler/scaler, trainable names, epoch and exact batch
cursor, Python/NumPy/Torch/CUDA RNG, resolved config, and provenance.
