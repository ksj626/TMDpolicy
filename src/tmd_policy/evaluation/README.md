# LIBERO evaluation

`PI05InferencePolicy` constructs no SmolVLA model. It uses official processing,
one immutable condition cache, deterministic seeded `[B,50,32]` noise, configured
PI0.5 Euler steps, official postprocessing, and returns `[B,50,7]`.

SmolVLA `sampler_mode: official` rejects overrides and checks the checkpoint
default is 10 steps. `sampler_mode: override` requires an explicit ablation label;
the shipped override is four steps. TMD uses the same outer/inner transition
conventions as training.

DMD2 checkpoints use their shifted denoise--renoise grid, starting from seeded
Gaussian noise. An explicit `--outer-steps 1` override evaluates the same
checkpoint as a one-step clean predictor rather than switching to Euler.

The environment loop uses `policy.device`, replans after the execution horizon,
and records immutable locator/revisions, step count, coordinate contract, and
seed rule. The reportable suite horizons are fixed to Spatial/Object/Goal/Long
`280/280/300/520`. Simulator RNG seed and fixed-init-state index are explicit;
evaluation trial `k` uses init state `k % 50`. Comparisons fail unless every arm has identical
`(suite,task_id,reset_seed)` keys, then report paired overall/suite/task results.

LIBERO-Plus runs in the separate `tmdpolicy-libero-plus` environment because its
fork replaces vanilla LIBERO. The evaluator verifies the classification mapping
and exact 10,030-task counts, creates only one task environment at a time,
appends each completed episode to `episodes.jsonl`, and supports exact
`--resume`. Final summaries include suite, perturbation category, and difficulty.
The baseline wrapper selects official SmolVLA-10, explicit frozen-checkpoint
SmolVLA-4/1 ablations, or official 10-step PI0.5 while retaining this same
LIBERO-Plus evaluation contract.
