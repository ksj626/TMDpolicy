# Short-window occupancy discriminator

This is separate from the DMD2/TMD conditional GAN. Its dataset unit is one
actual replan `(s_t,o_t,a_plan[t:t+49],task)`. Expert samples use the real chunk
start and valid future-action mask. Student samples use v2 raw replan cameras,
state, the full stored plan, executed-prefix metadata, behavior checkpoint, and
collection round. Reconstructed executed-action windows are forbidden.

Both classes are noised before classification. The default
`pi05_intermediate_features` variant uses official PI0.5 conditioning and
separate layer heads. `cached_vla_features` encodes SmolVLA condition once and
is explicitly an efficient VLA adaptation. Balanced-prior logits estimate the
joint short-window expert/student ratio; fixed historical rollout data is
documented as behavior occupancy, not exact current on-policy occupancy.

Training requires expert support = student support = configured support = all
40 global LIBERO tasks. Its train sampler pairs expert/student samples by task,
is a pure function of seed/epoch, and resumes from an exact batch cursor.

The optional `occupancy_tmd` weighting stage loads this v2 discriminator
immutably and evaluates each current full generated plan under its actual
current observation. It does not revive the retired repeated-start-state window
model. Detached clipped `exp(logit/temperature)` weights scale per-sample TM-MF
losses; provenance records the fixed behavior checkpoint and feature variant.
