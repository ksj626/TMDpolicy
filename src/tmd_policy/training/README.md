# Training engine

`engine.py` runs concrete `TrainingProgram` graphs on real PyTorch DataLoaders.
It supports BF16/FP16 autocast, gradient accumulation and clipping, separate
ordered optimizer phases, cosine decay, periodic validation, JSONL metrics, and
safe non-overwriting outputs. A terminal progress bar reports completed global
steps and current phase losses; validation has its own transient bar. The engine
also atomically replaces `training_progress.png` after every completed step with
a bounded-cost plot of all train and validation loss series.

Checkpoints contain the whole program, every optimizer/scheduler/scaler,
global step, sampler epoch and consumed-batch offset, Python/NumPy/Torch CPU and
CUDA RNG, resolved config, exact trainable names, and asset/source provenance.
Permutations are pure functions of seed and epoch; counters advance only after
a completed optimizer cycle, so resume at a saved boundary is exact. Default
`num_workers: 0` is the strictest reproducibility setting for video decoding.

Programs may additionally expose a minimal inference state. DMD2 writes its
trained SmolVLA delta periodically as `tmdpolicy.inference/v1`; the immutable
hub revision in the resolved config supplies all frozen weights. These small
files are valid for evaluation but intentionally invalid for exact training
resume, which still requires `tmdpolicy.training/v1`.

Multi-rate phase schedulers use actual optimizer counts. DMD2 therefore has a
500,000-update guidance schedule and 100,000-update generator schedule for a
100,000-global-step run, with correspondingly scaled warmup lengths. The same
counters, current learning rates, scheduler steps, phase/group gradient
diagnostics, backward-simulation trace, score/DMD/GAN diagnostics, and fixed
probe validation values are persisted in `metrics.jsonl`.

`TrainingProgram` defines phase/loss/optimizer hooks;
`DeterministicBatchSampler` owns epoch permutations; `seed_everything` owns
global seeds; `run_training` executes and checkpoints. `TrainingBundle` plus
`build_student`, `build_teacher`, `build_expert_datasets`, and
`build_training_bundle` construct every real method graph. `file_sha256`
validates dependent local checkpoints before any state is loaded.
