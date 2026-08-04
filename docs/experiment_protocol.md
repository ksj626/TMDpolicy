# Motivation and main protocol

Build data once, validate PI0.5 parity, train Flow-SFT and Stage-1, train DMD2 and
Stage-2 as resources permit, collect current-student rollout rounds, train the
occupancy discriminator, then train occupancy-TMD. Record each upstream SHA in
dependent configs. Never reuse validation/test episodes to fit normalization or
occupancy statistics. The shipped occupancy configuration is explicitly a
fixed-checkpoint, off-policy experiment.

Motivation uses four diagnostic tasks from each of four LIBERO suites and 20
shared reset seeds (320 complete episodes/arm). Main uses all ten tasks per suite
and 50 shared seeds (2,000/arm). Training seeds should be repeated independently
when estimating training variance; within one checkpoint comparison, environment
seeds and policy-noise derivation are paired exactly. Primary outcomes are macro
task success and paired difference from Flow-SFT; secondary outcomes are micro
success, per-task Wilson intervals, exact McNemar discordance, synchronized warm
replan latency, path smoothness, and occupancy discrimination diagnostics.

Do not inspect main results to tune hyperparameters. Mark static rollout reuse as
off-policy. A current-policy occupancy claim requires separately collecting a
new uniquely named rollout store from the current checkpoint, recording its
SHA, and training against that new store; the trainer does not refresh an
environment implicitly.
