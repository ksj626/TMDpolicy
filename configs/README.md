# Configuration reference

All experiment YAML is fully resolved and immutable-asset oriented.

Shared fields:

- `method`, `classification`: concrete builder identity and truthful fidelity label.
- `backend.lerobot_version`: exactly `0.6.1`; `expected_source_hashes` optionally
  pins the four imported module hashes; `local_files_only` forbids Hub fetches.
- `models.{student,teacher}.{id,revision,processor_revision}`: Hub identity and
  40-character immutable commits.
- `dataset.id`, `revision`, `cache`, `manifest`: `lerobot/libero`, immutable
  commit, repo-local cache, and episode manifest. Fractions/seed define the
  task-stratified episode split. Video fields configure the real reader.
- `horizons.prediction/execution`: `[50,10]` by default.
- `training`: seed, device, batch/workers, precision, accumulation, clipping,
  AdamW hyperparameters, warmup/cosine floor, maximum optimizer cycles,
  validation/checkpoint intervals, and validation batch limit.
- `output.directory`: safe default; CLI `--output` may override it.

Method fields:

- `fine_tuning`: Flow-SFT mode (`head_only`, `expert_only`, `lora`, `full`) and
  LoRA rank/alpha/dropout.
- `tmd`: head variant, `r=s` fraction, adaptive constant, early/last expert
  split, transformer/GRU dimensions, dropout, and inference steps.
- `dmd2`: student mode, fake variant, TTUR, shared generation steps, optimizer
  rates, GAN/data weights, explicit score-time interval, device placement,
  network dimensions, and resource estimate.
- `stage1_architecture`/`stage2`: exact Stage-1 reconstruction and checkpoint
  SHA, shared sampler steps, then DMD2-v fields.
- `rollouts`: versioned store and window stride.
- `occupancy_model`: window/model dimensions and train-only normalization cap.
- `occupancy`: discriminator checkpoint/SHA, density-ratio clipping/temperature,
  weight-sampler steps and the explicit fixed-checkpoint off-policy rollout mode.
- `parity`: device/dtype/fixed steps/noise seed/minimum score time.
- `policy`: inference method, checkpoint/SHA, device, outer/inner steps.
- `collection`: suite/tasks, disjoint reset-seed splits, horizon, and round.
- `evaluation`: suite/task matrix, paired reset seeds, episode horizon, rollout
  saving, and CUDA synchronization. Comparison bootstrap count/confidence/seed
  live under `statistics` in the experiment config.
- `inputs`/`statistics`: named evaluation JSON paths (including `baseline`),
  exact pairing keys, bootstrap resample count, confidence, and RNG seed.

The Stage-1 and occupancy checkpoint SHA placeholders must be replaced after
their upstream run. No `lerobot_commit`, executability declaration, or synthetic
research runner field exists.
