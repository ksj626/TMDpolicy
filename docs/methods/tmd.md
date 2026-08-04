# Transition Matching Distillation action-space contract

Classification: **paper-faithful action-space port**. It is not an exact
architecture reproduction: the video paper reuses final Wan transformer
blocks, whereas SmolVLA's action expert cannot be partitioned at the identical
block boundary without changing the pinned model. The port uses a separately
checkpointed recurrent action-flow head conditioned on frozen/cached main
backbone features. The former `gaussian_tm` implementation is renamed
`tmd_plain_gaussian_ablation` and is not full TMD.

Primary equations are from Transition Matching (arXiv:2506.23589, Eqs. 8–13)
and TMD (arXiv:2601.09881v2, Eqs. 1–18 and Algorithms 1–2).

## Outer transition and inner variables

For data `x`, outer noise `x1~N(0,I)`, and `0=t0<...<tM=1`:

```text
x_t = (1-t)x + t x1                 (TMD Eq. 1)
y   = x1 - x                        (Eq. 3)
x_t(i-1) = x_t(i) - (ti-ti-1)y     (Eq. 4)
y_s = (1-s)y + s y1, y1~N(0,I)     (Eq. 5)
v(y_s,s) = y1-y                     (conditional velocity)
```

All action tensors are `[B,H,D]`; `t,s,r` are `[B]` and broadcast. `x1`, `y1`,
and sampled times are independent. Unlike the retired shortcut, `y1` is not a
prediction-head input. The head sees `y_s`, `s`, `r`, and main feature `m`.

## Stage 1: TM-MeanFlow

```text
f(y_s,s,r;m) = y_s + (s-r) u_theta(y_s,s,r;m)       (printed Eq. 13)
u_theta = y1 - head_theta(y_s,s,r;m)                 (Eq. 14)
u_target = sg(v(y_s,s) - (s-r) d/ds u_theta)         (Eqs. 9–10)
L_MF = ||u_theta-u_target||² / sg(||...||²+c)
```

The total derivative includes explicit `s,r` dependence and the path tangent
`dy_s/ds=v`. `meanflow_jvp` uses forward-mode JVP. The separately selected
paper finite difference uses central differences
`[u(y_{s+δ},s+δ,r)-u(y_{s-δ},s-δ,r)]/(2δ)` with
`y_{s±δ}=y_s±δv`, and one-sided boundaries. `r=s` is sampled for the configured
paper fraction (default 0.75). The target is stop-gradient; main/head gradients
follow the configured port, and are recorded.

There is an apparent sign inconsistency in the paper source: its interpolation
`y_s=(1-s)y+s*y1` and velocity `y1-y` imply that a more-noisy state at `s`
must map to a less-noisy state at `r<s` with `y_r=y_s-(s-r)u`. The printed plus
sign instead moves toward greater noise and fails the constant-velocity
endpoint check. No official TMD code was available at the source-review date.
This action-flow adaptation therefore uses the endpoint-consistent subtraction,
tests that endpoint analytically, and does **not** claim exact code
reproduction. The deviation remains explicit pending author code or erratum.

## Stage 2: DMD2-v

Inner flow is unrolled for every outer transition and
`g_theta(x_t,ti;y1)=x1-INNERFLOW(m_theta(x_t,ti))` (Eq. 15). Stage 2 applies
the separately implemented DMD2-flow VSD and GAN objectives to this output;
fake-score and discriminator updates occur between student updates. Gradients
backpropagate through every inner step. If a frozen teacher score/velocity is
not available, Stage 2 raises `MissingCapabilityError`; no MSE replacement is
permitted.

Sampling runs outer `i=M..1`, inner `j=N..1`; both `dt` directions are from
noise to data. Main backbone is evaluated once per outer step, head exactly `N`
times per outer step. All noise tensors can be supplied for replay.
