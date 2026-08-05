# DMD2-flow

The paper config uses the direct DMD2-v real-data outer transition, frozen
PI0.5 real model, PI0.5-initialized fake suffix, 5:1 joint guidance updates, DMD2's
teacher-residual-weighted distribution gradient, and a noised-action discriminator on fake-score layers
`[5,11,17]`. The generator objective is

```text
L_G = L_VSD + lambda_GAN softplus(-D(fake_tau))
```

and never calls `flow_matching_loss`. The discriminator averages separate-layer
`softplus(-D(real_tau))+softplus(D(fake_tau))` losses. Its generator path freezes
feature/head parameters but preserves and checks a nonzero fake-action gradient.

Training samples nonzero `t_i` from the shifted student grid, forms
`x_t=(1-t_i)x+t_i*x1`, and returns `x_hat=x_t-t_i*v_student(x_t,t_i)`. Evaluation
uses deterministic Euler integration over that same checkpoint-owned grid. This
is the TMD paper's direct DMD2-v baseline, not original DMD2 backward simulation.
Each fake-feature guidance update reuses one detached student sample and minimizes
`L_fake + guidance_classifier_weight*L_classifier` with one optimizer containing
disjoint fake-score and classifier parameter groups.

`pi05_clone` shares the teacher prefix and independently trains the action
expert/projections. `smolvla_clone` and `lightweight` remain code-level ablations
but cannot be selected by a baseline config. `cached_vla_features` is a named VLA
adaptation and is not the paper-feature default.

Tensor contract is internal `[B,50,32]`; only seven executable dimensions and
valid timesteps affect score, normalization, and loss metrics. Provenance records
feature source (`fake_score_features` for DMD2), layers, devices, parameter counts,
time shifts, and the explicit absence of SFT.

For DMD2, let `x` be the generated action and `mu_real`, `mu_fake` be the
teacher/fake denoised predictions. Its stopped direction is
`(mu_fake-mu_real)/(mean_valid(abs(x-mu_real))+epsilon)`, matching the reference
implementation's `p_real` weighting. This is intentionally different from the
TMD-v valid-L1 normalization.
