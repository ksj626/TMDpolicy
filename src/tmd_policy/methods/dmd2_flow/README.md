# DMD2-flow

`program.py` generates clean student chunks with the same multi-step
`LeRobotSmolVLAStudent.sample` used at inference, corrupts those generated
chunks dynamically, queries the online frozen PI0.5 score at those points,
trains a fake score with the configured TTUR, trains a task/state-aligned GAN
critic on real expert versus generated chunks, and retains an expert data loss.
The student-to-teacher normalization path remains differentiable; PI0.5 outputs
are detached. Real labels are 1 and generated labels are 0.

`fake_scores.py` provides `pi05_clone` (closest but extremely expensive) and
`smolvla_clone` (cross-architecture adaptation). `networks.py` provides the
default lightweight action-score transformer, explicitly non-paper-faithful.
All models use `[B,50,32]` normalized flow tensors; fake-score/DMD
normalization and GAN actions use only the seven valid dimensions plus terminal
masks. The frozen teacher and a clone fake score may remain on explicitly
separate GPUs when the training program is placed on the student GPU.
Checkpoints include all trainable networks, optimizers, schedules, RNG, sampler
cursor, and source/asset provenance.

`DMD2FlowProgram` is the alternating program. `PI05CloneFakeScore` and
`SmolVLACloneFakeScore` are explicit clone variants;
`ActionScoreTransformer` and `ActionChunkDiscriminator` are the lightweight
conditioned networks. Flow/action tensors are float32, autocast governs model
activations, corruption uses checkpointed Torch RNG, and only generator-owned
coordinate transforms retain gradients.
