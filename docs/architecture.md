# Architecture and tensor contract

The dependency direction is data/evaluation → stable backend protocols →
method program → shared training engine. Algorithms cannot call arbitrary
LeRobot internals. The compatibility layer validates the exact 0.6.1 signatures
it consumes and hashes their source files. Official pre/postprocessors own
images, state, instruction tokens, and checkpoint normalization.

PI0.5 caches image/language prefix embeddings and KV values once per observation.
The typed cache records batch/prefix dimensions, model and processor revisions,
dtype/device, and coordinate contract. Each suffix query uses the installed
one-step `denoise_step`; tensor storage/shape/version fingerprints prove the
retained prefix was not extended. Teacher parameters are frozen and outputs are
detached. The explicit score floor is configured, never implicit.

Canonical action `a` is float `[B,50,7]`; `True` mask `[B,50]` is valid. Student
and teacher normalizers are reconstructed from loaded processor mode and tensor
statistics. The bridge evaluates `N_T(N_S^{-1}(z_S))`, pads zeros to
`[B,50,32]`, and masks terminal/extra dimensions without CPU or NumPy, preserving
the generator gradient. Relative/ALOHA semantics fail closed.

The real dataset reader requests offsets `0..49` at dataset FPS. LeRobot repeats
the terminal boundary and emits `action_is_pad`; losses use its inverse. Whole
episodes are task-stratified before reader construction. Rollout windows contain
executed canonical states/actions/visual features and producer identity.

The engine treats one DMD optimizer cycle as configured fake-score updates,
one discriminator update, and one generator update. It saves only after complete
cycles. Deterministic epoch permutations plus consumed-batch offsets, complete
RNG, optimizer/scheduler/scaler, and module state make saved-boundary resume
equivalent to uninterrupted execution.
