# Occupancy TMD

`discriminator.py` is a causal `[B,L+1,S]`/`[B,L,A]` path classifier distinct from DMD2 GAN. `weights.py` separates clipped prioritization from valid importance ratios/ESS. `method.py` provides isolated discriminator BCE training and held-out-gated downstream weighting; the discriminator is frozen during TMD. See the method contract for balanced-measure ratio limits and diagnostics.

`OccupancyTMDConfig` defaults are weights `[0.5,2.0]` and discriminator LR
`1e-4`. Gate thresholds are source deviation `0.05`, calibration `0.1`,
saturation `0.1`, support overlap `0.5`, and ESS `20`, all from real held-out
paths. Diagnostic optimizer state is separate; downstream occupancy-TMD freezes
the discriminator and does not conflate its weights with importance ratios.
