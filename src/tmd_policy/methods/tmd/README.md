# TMD

`meanflow.py` implements Stage-1 action MeanFlow variables `s,r`, average velocity, JVP/finite difference, stop-gradient target, and inner noise-to-transition rollout without exposing source noise to the head. `method.py` isolates Stage 1 and capability-gates Stage 2. Full equations, the printed Eq. 13 sign discrepancy, gradients, defaults, architecture port, and limitations are in `docs/methods/tmd.md`.

`MeanFlowConfig` defaults are dimensions `7/64/128`, finite-difference delta
`0.005`, `r=s` fraction `0.75`, adaptive constant `350`, and derivative mode
`finite_difference` (`jvp` is separate). Stage 1 owns backbone/head and one
optimizer/scheduler. Stage 2 stores the TMD generator, fake score, DMD2 GAN,
three optimizer streams, counters, and immutable Stage-1 identity.
