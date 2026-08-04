# Official SmolVLA Flow-SFT contract

Classification: **exact reproduction of the pinned LeRobot SmolVLA training
objective**, with repository-level additions for per-sample masking, resume,
and validation. It is not a reproduction of a separate paper.

## Equation and variables

For normalized expert action chunks `A ∈ R[B,H,D]`, independent
`epsilon ~ N(0,I)`, and the official beta-sampled `t ∈ (0,1)`:

```text
A_t = (1-t) A + t epsilon
v*  = epsilon - A
L_i = sum_{h,d} M[i,h] (v_theta(A_t,t,O)-v*)²
      / (D sum_h M[i,h])
L = mean_i L_i
```

`O` is the officially processed image/state/language observation. `H=50` and
canonical `D=7` before internal padding. `M[B,H]` is boolean and every sample
must contain at least one valid action. Actions are normalized and
postprocessed only by the pinned SmolVLA processor/normalizer revisions.

Sampling starts at `A_1=epsilon` and integrates `dA_t/dt=v_theta` from `t=1`
to `0`, hence every Euler `dt` is negative. The returned internal chunk is
passed through the official postprocessor.

## Gradients and algorithm mapping

`flow_sft_loss` implements the pinned model's interpolation and velocity MSE.
The observation processor and normalization statistics are constants.
Backbone modes are isolated: `frozen`, `lora`, or `full`; the checkpoint lists
the exact trainable names. Gradient accumulation delays optimizer/scheduler
updates, while mixed precision only changes numerical representation.
Validation reuses fixed `epsilon,t` tensors and never advances training RNG.

## Action-flow transfer

The equation, schedule, representation, masking, and sampling direction
transfer exactly because SmolVLA is already an action flow. Episode-disjoint
splitting and deterministic checkpointing are framework additions, not changes
to the objective.
