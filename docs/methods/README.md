# Fidelity labels

Flow-SFT uses the exact official SmolVLA objective. The split-transformer TMD
head is the primary paper-closest action-space port but is not an exact
paper-native architecture because LeRobot exposes no supported partial expert
forward; the GRU is an explicit lightweight adaptation. DMD2 with `pi05_clone`
is closest to the original fake-score architecture; SmolVLA/lightweight variants
are adaptations. Stage-2 uses the actual Stage-1 sampler. Occupancy-TMD is a
proposed extension. Source-level equations, tensor shapes, masks, gradient
ownership, and limitations are documented in each `src/tmd_policy/methods/*`
README.
