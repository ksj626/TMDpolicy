# Transition Matching Distillation

`meanflow.py` samples independent outer/inner Gaussian sources, samples
`0≤r≤s`, makes the configured Bernoulli fraction exactly use `r=s`, computes
the total derivative with a JVP, stops gradients only through the constructed
target, applies coordinate masks, and uses adaptive per-sample normalization.
`heads.py` contains the primary `tmd_split_transformer_head` port and the
explicitly non-paper-faithful `tmd_gru_head`. `program.py` couples those equations
to real SmolVLA features/actions and the resumable trainer.

All action tensors are normalized `[B,50,32]`; only the first seven dimensions
and nonterminal timesteps contribute. The early action expert is evaluated once
per outer state. LeRobot 0.6.1 offers no supported partial-layer forward, so the
repository-owned split transformer is the inner flow head. This is the closest
supported SmolVLA architectural port, not an exact reproduction of the paper's
native backbone. Its descending inner Euler convention is shared by training,
Stage 2, and inference.

Public math helpers are `sample_meanflow_batch`, `meanflow_total_derivative`,
`meanflow_loss`, and `integrate_inner_flow`; `MeanFlowBatch` records all sampled
sources/times. `SplitTransformerMeanFlowHead` is primary and
`GRUMeanFlowHead` adapted. `TMDStage1Program`, `sample_stage1_generator`, and
`TMDStage2Program` own training/shared sampling/checkpoint refinement. Gaussian
sources use Torch RNG on the action device; teacher results and MeanFlow targets
are detached at their documented boundaries.
