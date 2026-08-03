# Gaussian transition-matching method report

## Outer flow

Let `A ∈ R^{B×50×D}` be a student-normalized action chunk and draw
`epsilon~N(0,I)` with identical shape, dtype, and device. Training samples
`t∈(0,1]` and defines

```text
A_t = (1-t)A + t epsilon,   Y = dA_t/dt = epsilon-A.
```

Sampling begins at `A_1=epsilon`, evaluates a velocity, and uses Euler
`dt=-1/N`. The analytic oracle field is constant, so any positive `N` ends at
`epsilon + (-1)(epsilon-A)=A`. Tests check the sign, endpoint, and wrong-sign
counterexample.

## Default Gaussian inner flow

For each outer evaluation the cached SmolVLA backbone runs once and returns
reference `B` and action-token features `F`. Independently draw
`Z~N(0,I)`, `Z.shape=Y.shape`. The analytic inner path is

```text
Y_s = (1-s)Y + sZ,          dY_s/ds = Z-Y,          s:1->0.
```

The recurrent head sees current inner state, fixed source `Z`, reference `B`,
outer state/time, inner time, position, and `F`. It predicts a residual:

```text
Y_hat = B + Delta_theta(...),       u_theta = Z-Y_hat.
```

Training compares `u_theta` literally with `Z-Y`. Algebraically the error is
`Y-B-Delta`, but retaining the velocity construction in code makes source/sign
auditing direct. Valid coordinates are reduced over horizon/action dimensions
per sample, then over recurrent inner evaluations. `reduction='none'` returns
`[B]`; only `reduction='mean'` averages the batch.

Inference starts `current=Z` and repeats
`current <- current + (-1/M) u_theta` at descending `s`. If the zero-initialized
output gives `Delta=0`, the constant field `Z-B` returns `B` exactly. If an
oracle gives `Delta=Y-B`, the constant field `Z-Y` returns `Y` exactly. CPU/CUDA
tests cover both for multiple inner-step counts and empirically check standard
normal source statistics.

Each inner evaluation carries GRU hidden state across `s`; the outer loop draws
one distinct `Z` per expensive call. Fixed outer and all inner noises reproduce
actions exactly and counters must equal `N` backbone plus `N×M` head calls.
Inference APIs do not accept target actions.

## Modes and pretrained preservation

`gaussian_tm` is the default and only primary method. The historical prototype
is retained as `anchored_tm_ablation`: it begins at `B`, trains a direct
anchor-to-`Y` velocity, and zero initialization is a no-op. It is not reachable
through missing/default/short aliases. `gaussian_tm_meanflow` is reserved and
raises an explicit unsupported-mode error.

The vision/language/base model is frozen. By default only the transition head is
trainable. An explicit config may also enable the small action input/output,
action-time, and state projections; those names and tensors are checkpointed.
The discriminator and teacher are frozen during student updates, weights are
detached, and environment/storage tensors never enter an autograd graph.

## Causal discriminator and weighting semantics

A path token concatenates normalized `(s_j,a_j,s_{j+1}-s_j,s_{j+1})`, then adds
task and position embeddings. The prefix variant applies a strict causal mask;
pointwise and final-only variants provide controlled baselines. Expert is label
1, exact current is label 0, and equal effective class priors plus task/position
balanced BCE give the intended approximate orientation
`log rho_E/rho_current`. Calibration metrics accompany classification metrics.

The local increment `r_j=l_j-l_{j-1}`, `l_-1=0`, is an estimated conditional
log-ratio change under finite model/calibration—not an automatically exact
reward. `MismatchWeights` are bounded detached emphasis for low expert-likeness.
They are type-separated from `ImportanceWeights`, which implement the explicitly
oriented replay identity

```text
log rho_E/rho_C = log rho_E/rho_B + log rho_B/rho_C.
```

Historical replay cannot replace exact `fresh_current` samples. Teacher cache
records are a third immutable pool.

## Scope relative to TMD

The transition head is appended to SmolVLA action-token features rather than
repurposing transformer blocks, keeping the installed LeRobot source unchanged.
This code studies action-flow repair on LIBERO, not video generation. It excludes
adaptive solvers, DMD2/fake-score methods, a implemented MeanFlow objective, and
claims of full occupancy RL. B3/B4 teacher distillation remains gated until
complete-episode B0/B1/B2 evidence is saved.
