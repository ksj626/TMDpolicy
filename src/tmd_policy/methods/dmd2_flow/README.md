# DMD2-flow

`losses.py` defines the score-difference generator field, masked normalization, fake-score regression, and a distinct conditional action GAN. `method.py` owns generator/fake-score/GAN models, three optimizers/schedulers, 5:1 TTUR, inference-input simulation, freezing, sampling, and complete checkpoint state. See `docs/methods/dmd2_flow.md` for equations and score assumptions.

`DMD2Config` defaults are fake-score updates `5`, three learning rates `1e-5`,
GAN weight `3e-3`, score support `[1e-3,0.999]`, and generation schedule
`(0.999,0.749,0.499,0.249)`. Inputs are `[B,H,D]`, `[B,C]`, `[B,H]` plus
explicit replay noises/times. Every model, optimizer, scheduler, counter,
config, trainable name, and RNG is checkpointed.
