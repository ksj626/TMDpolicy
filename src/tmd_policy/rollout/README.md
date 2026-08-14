# LIBERO replan storage

Schema `tmdpolicy.libero-replans/v2` stores one payload per episode and one
record per actual policy replan. Every record includes suite/local/global task
identity, canonical UID/instruction, reset seed, behavior checkpoint and SHA,
fixed-init-state index, policy version/round, environment step, exact state, lossless camera tensors and
shape/dtype/layout metadata, full canonical `[50,7]` plan, executed prefix and
actions, terminal/truncation/success state, and immutable revisions.

Only observations returned by LIBERO are stored. Full plans are never rebuilt
from executed actions and future states are never synthesized. Payload and index
writes use temporary files plus atomic replacement. v1 episode-summary stores
cannot be migrated implicitly and fail with a recollection message.
