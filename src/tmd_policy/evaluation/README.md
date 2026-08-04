# LIBERO evaluation

`PI05InferencePolicy` constructs no SmolVLA model. It uses official processing,
one immutable condition cache, deterministic seeded `[B,50,32]` noise, configured
PI0.5 Euler steps, official postprocessing, and returns `[B,50,7]`.

SmolVLA `sampler_mode: official` rejects overrides and checks the checkpoint
default is 10 steps. `sampler_mode: override` requires an explicit ablation label;
the shipped override is four steps. TMD uses the same outer/inner transition
conventions as training.

The environment loop uses `policy.device`, replans after the execution horizon,
and records immutable locator/revisions, step count, coordinate contract, and
seed rule. Comparisons fail unless every arm has identical
`(suite,task_id,reset_seed)` keys, then report paired overall/suite/task results.
