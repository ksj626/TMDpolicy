# VLA-OPD contract

Source: VLA-OPD arXiv:2603.26666v1, Eqs. 3–7 and Algorithm 1. The project page
listed official code as unavailable on 2026-08-04.

## Exact categorical method

Classification: **exact objective reproduction** for policies exposing
normalized categorical token probabilities. For current-policy trajectory
states `s_t`, action tokens `a_t~pi_theta`, and frozen teacher `pi_tea`:

```text
r_t = -stopgrad(log pi_theta(a_t|s_t)-log pi_tea(a_t|s_t))  (Eq. 6)
J gradient = (1/G) sum_i sum_t grad log pi_theta(a_ti|s_ti) r_ti  (Eq. 7)
```

The raw reward is used; there is no GRPO outcome normalization. `G` fresh
trajectories are executed by the exact current policy for each prompt. The
teacher labels every visited state but its actions are never executed.
Historical states cannot satisfy a new exact round.

## Continuous-flow port

Classification: **paper-faithful continuous-flow adaptation**, separately
named `continuous_flow_opd`. A deterministic flow from base Gaussian `z` to
action chunk `a` has

```text
log p(a|O) = log N(z;0,I) - integral_0^1 div_x v_theta(x_t,t,O) dt.
```

The exact mode computes the full Jacobian trace. An optional Hutchinson mode is
explicitly a stochastic trace estimator with recorded probe seeds. Teacher and
student log densities must use the same canonical action measure, including
normalizer Jacobians. The OPD reward and stop-gradient are then unchanged.

The pinned public pi0.5 policy has sampling and an internal denoise step but no
supported `log_prob`, inverse flow, or conditional-density API. Therefore the
default pi0.5 continuous OPD capability check fails closed. The generic CNF
density implementation is tested on analytic flows and can be enabled only by
a future audited adapter; velocity MSE is never substituted.

Every record stores collection round, exact policy version, old/current policy
identity, base-noise seed, student action, both log densities, divergence mode,
integration schedule, and teacher capability revision.
