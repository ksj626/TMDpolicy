# VLA-OPD

`losses.py` implements exact categorical reverse-KL policy gradient and the separately named continuous-flow CNF port with detached reward/action. `method.py` enforces fresh policy version/round groups, freezes the teacher, and checkpoints student/optimizer/round. Pinned pi0.5 fails closed because its public API supplies neither normalized probabilities nor supported density at a student action.

`OPDConfig` defaults are group size `8`, learning rate `1e-6`, CNF steps `32`,
divergence `exact`, and strict freshness. Categorical logits are `[G,T,V]` and
continuous actions `[G,H,D]`. Checkpoints contain student/frozen teacher,
optimizer/scheduler, round/current policy version, config, names, and RNG.
Hutchinson mode requires an explicit replayable probe.
