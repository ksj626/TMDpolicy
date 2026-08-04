# Training engine

`engine.py` runs concrete `TrainingProgram` graphs on real PyTorch DataLoaders.
It supports BF16/FP16 autocast, gradient accumulation and clipping, separate
ordered optimizer phases, cosine decay, periodic validation, JSONL metrics, and
safe non-overwriting outputs.

Checkpoints contain the whole program, every optimizer/scheduler/scaler,
global step, sampler epoch and consumed-batch offset, Python/NumPy/Torch CPU and
CUDA RNG, resolved config, exact trainable names, and asset/source provenance.
Permutations are pure functions of seed and epoch; counters advance only after
a completed optimizer cycle, so resume at a saved boundary is exact. Default
`num_workers: 0` is the strictest reproducibility setting for video decoding.

`TrainingProgram` defines phase/loss/optimizer hooks;
`DeterministicBatchSampler` owns epoch permutations; `seed_everything` owns
global seeds; `run_training` executes and checkpoints. `TrainingBundle` plus
`build_student`, `build_teacher`, `build_expert_datasets`, and
`build_training_bundle` construct every real method graph. `file_sha256`
validates dependent local checkpoints before any state is loaded.
