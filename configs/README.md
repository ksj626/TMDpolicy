# Configuration reference

All assets require immutable 40-character Hub revisions and LeRobot `0.6.1`.
Unknown method variants and deprecated faithful-objective fields fail before
model loading. In particular, `data_weight` is an error for `dmd2_flow` and
`tmd_stage2`.

Faithful method configs:

- `dmd2_flow_paper.yaml`: `pi05_clone`, fake-score layer features, no SFT.
- `tmd_stage1_action_head.yaml`: bidirectional action head, corrected base
  preconditioning, shifted/discrete TM-MF.
- `tmd_stage2_paper.yaml`: automatic immutable Stage-1 SHA, `pi05_clone`, teacher
  layer features, real-data outer transitions, no SFT.
- `occupancy_discriminator_paper.yaml`: all-40-task v2 replans and PI0.5 features.

Named adaptations are `dmd2_flow_cached_vla_ablation.yaml`,
`tmd_stage2_cached_vla_ablation.yaml`, and
`occupancy_discriminator_cached_vla.yaml`. They never masquerade as paper
feature defaults.

Time sections `vsd_time`, `gan_time`, and `fake_score_time` require
`0 <= minimum_time < maximum_time <= 1` and `time_shift_gamma >= 1`. TMD Stage 1
uses `student_time_shift_gamma` for both discrete training/inference grids and
`meanflow_time_shift_gamma` for continuous `s`. `discrete_outer_steps` and
`discrete_inner_steps` are checkpoint architecture fields. Nonzero
`condition_dropout_probability` is rejected because this conditional baseline
has no CFG. `normalization_constant_scale: 1.0` yields `c=350` for `[50,7]`.

`discriminator.feature_source` is explicitly `fake_score_features` for DMD2 and
`teacher_features` for TMD Stage 2. Selected layers and separate-head aggregation
are stored in provenance. Cached condition identity includes encoder/processor
revision, layer, dtype, dimension, components, and detach policy.
DMD2 also records `student_training_mode: real_data_outer_transition` and a
separate `guidance_classifier_weight`; Stage 2 records
`student_training_mode: tmd_stage1_outer_transition` and an explicit
`discriminator_updates_per_generator`.

`vsd_normalization` is method-locked: DMD2 uses
`dmd2_teacher_residual_mean_abs`, while TMD Stage 2 uses
`tmd_fake_teacher_difference_l1`. A config cannot silently exchange them.

Checkpoint hashes may be `auto`: the exact file SHA-256 is computed, validated,
inserted into the in-memory config, and written to the run's resolved YAML before
training/evaluation. There are no unresolved placeholder hashes.

Evaluation configs distinguish `pi05`, SmolVLA `official`, and SmolVLA
`override`. Official SmolVLA forbids `num_steps`; override requires an ablation
classification. Collection configs must enumerate the four suites and every
local task 0–9. Occupancy configs must enumerate global task indices 0–39.

`preflight.minimum_total_memory_gib` is a per-device hard floor. A failed
preflight reports the requested component device and detected memory and never
substitutes a smaller score/discriminator.
