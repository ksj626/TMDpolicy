# Transition Matching Distillation

Stage 1 samples real `x`, outer `x1`, shifted discrete outer time, transition
`y=x1-x`, and independent inner `z`. Inner `s` is shifted; for non-FM rows `r`
is the largest shifted inner-student-grid point with `r<=s`, while `r=s` on the
configured fraction. The SmolVLA base velocity
is retained:

```text
b = v_SmolVLA(x_t,t,c)
h = b + Delta(y_s,s,r,m)
u = z - h
```

The exact JVP implements MeanFlow and adaptive normalization
`sum(e^2)/(stopgrad(sum(e^2)+scale*N_valid))`; padding and non-executable
coordinates are masked, so scale `1.0` yields `350` for `[50,7]`. Both training
and inference use the same checkpoint-owned shifted student grids. This
conditional action-policy adaptation has no CFG. Bidirectional action-token attention matches the
and non-executable coordinates. Bidirectional action-token attention matches the
joint SmolVLA action expert; causal is an explicit ablation. Zero `Delta`
reproduces the SmolVLA transition for 1/2/4 inner steps.

Stage 2 loads the immutable Stage-1 checkpoint and asserts the exact final-K
expert-block trainability set survived construction. Each update builds
`x_t=(1-t)x+t*x1` from real data, computes one main feature/base velocity,
unrolls every inner step from independent `z`, returns `x_hat=x1-InnerFlow`, and
applies VSD plus teacher-feature GAN. No Flow-SFT/data loss is present.

This is a repository-owned SmolVLA action-space adaptation, not an exact native
split of the video model used by the paper. Architecture fidelity and objective
fidelity are labeled separately in configs and checkpoints.
