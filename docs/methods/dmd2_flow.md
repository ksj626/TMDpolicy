# DMD2-flow action-space contract

Classification: **paper-faithful action-flow port**, not an exact image-model
architecture reproduction. Source: DMD2 arXiv:2405.14867v2 and official
`tianweiy/DMD2` training code.

## Score and distribution-matching gradient

For a forward corruption `x_t=alpha_t x + sigma_t epsilon`, the paper's score
identity is

```text
s_p(x_t,t) = -(x_t-alpha_t E_p[x|x_t]) / sigma_t².       (Eq. 1)
```

For the SmolVLA rectified schedule, `alpha_t=1-t`, `sigma_t=t`. If a velocity
oracle supplies `v_p(x_t,t)=E[epsilon-x|x_t]`, then

```text
E[x|x_t] = x_t - t v_p(x_t,t)
s_p(x_t,t) = -(x_t + (1-t) v_p(x_t,t))/t.
```

This conditional-expectation identity is valid only for the stated Cond-OT
schedule. It is singular at `t=0`; the support is explicitly
`t~Uniform[t_min,t_max]` with paper-configured `t_min>0`, never silently
clamped.

The generator gradient follows DMD2 Eq. 2 using frozen real score and an
online fake score:

```text
g = -w(t) (s_real(x_t,t)-s_fake(x_t,t)) dG_theta/dtheta
```

The implementation uses a stop-gradient pseudo-target to realize this vector
Jacobian product and the official per-sample stabilization denominator. Fake
score denoising loss is evaluated on detached current generator actions.

## TTUR and conditional GAN

The fake score is updated `K=5` times per generator update by default. Fake
score and generator have separate optimizers/schedulers. The DMD2 discriminator
is a distinct conditional action-chunk discriminator. For task-matched real
expert `x` and generated `G(z)` conditioned on the same observation/task:

```text
L_D = E softplus(-D(F(x,t),c)) + E softplus(D(F(G(z),t),c))
L_GAN,G = E softplus(-D(F(G(z),t),c)).
```

Real/fake priors are matched. Discriminator/fake-score parameters are frozen
during generator updates; generator outputs are detached during their updates.

## Multi-step generation

A fixed schedule is shared by training and inference. Training simulates every
inference-time intermediate input using the current generator and independent
noise; earlier simulation steps are detached unless backward simulation is an
explicit, separately named setting. Checkpoints include generator, frozen-
teacher identity, fake score, GAN discriminator, three optimizers/schedulers,
schedule, counters, RNG, and provenance.

No run may start unless the teacher exposes a velocity/score satisfying the
schedule contract. A sample-only teacher is insufficient.
