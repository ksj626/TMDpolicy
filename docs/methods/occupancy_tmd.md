# Proposed short-window occupancy-weighted TMD contract

Classification: **proposed method**. It is neither paper TMD nor DMD2's GAN.

For window length `L`, the causal discriminator receives

```text
(task/instruction, s0,a0,s1,...,a[L-1],sL)
```

and emits prefix logits `ell_j` using only tokens through `s_j`. Training uses
matched expert and exact-current student paths, identical preprocessing,
task/position-balanced sampling, and episode-disjoint splits:

```text
L_D = 1/2 E_qE softplus(-ell_L) + 1/2 E_qS softplus(ell_L).
```

At the balanced optimum, `ell=log(dqE/dqS)` for the *reweighted training
measures* `qE,qS`; it is not automatically a raw environment occupancy ratio.
Prefix increments `delta_j=ell_j-ell_{j-1}` have a conditional-ratio
interpretation only if both path measures admit the same causal factorization,
support, and matched prefix priors.

Detached clipped prioritization weights are a sampling heuristic and use the
`MismatchPrioritizationWeight` type. Mathematically valid replay importance
ratios use the distinct `ImportanceRatio` type and require stored behavior and
current-policy log densities. Neither is silently substituted for the other.

The gate requires real held-out calibration, source-only chance controls,
task/position controls, support overlap, non-saturation, effective sample size,
fresh-versus-replay and clipping diagnostics. Synthetic AUC cannot open it.
The distillation target remains the capability-checked TMD Stage 2 objective;
the discriminator only changes sampling/weighting.
