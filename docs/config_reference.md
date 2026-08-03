# Generated configuration reference

This file is generated from `tmd_policy.config`; edit the dataclasses, not this file.

| Field | Type | Default | Valid values | Mathematical meaning | Consumer | Cache? | Checkpoint? | Affects | Recommended |
|---|---|---|---|---|---|---|---|---|---|
| name | str | "tiny_libero_tmd" | nonempty filesystem-safe name | Run identity used for artifact directories. | cli._audit; config.save_resolved_config | no | no | storage | tiny_libero_tmd |
| horizons.prediction_horizon | int | 50 | exactly 50 for LIBERO | Number of actions in each planned chunk H_plan. | data.schemas.ExpertChunk; rollout.collector.collect_rollout_episode | yes | yes | training,inference,storage | 50 |
| horizons.execution_horizon | int | 10 | 1..prediction_horizon; exactly 10 for LIBERO | Maximum real environment transitions before replanning H_exec. | data.schemas.RolloutChunk; rollout.collector.collect_rollout_episode | yes | yes | training,evaluation,storage | 10 |
| canonical.state_dim | int | 8 | exactly 8 | Canonical LIBERO low-dimensional state width. | compatibility.actions.StateCompatibilityAdapter; data.schemas | yes | yes | training,inference,storage | 8 |
| canonical.action_dim | int | 7 | exactly 7 | Canonical LIBERO action width. | compatibility.actions.CanonicalActionSpace; data.schemas | yes | yes | training,inference,storage | 7 |
| canonical.action_min | float | -1.0 | exactly -1.0 | Inclusive lower bound after official postprocessing. | compatibility.actions.CanonicalActionSpace; data.schemas._canonical_actions | yes | no | inference,storage | -1.0 |
| canonical.action_max | float | 1.0 | exactly 1.0 | Inclusive upper bound after official postprocessing. | compatibility.actions.CanonicalActionSpace; data.schemas._canonical_actions | yes | no | inference,storage | 1.0 |
| checkpoints.student_id | str | "lerobot/smolvla_libero" | nonempty Hub ID or local path | Frozen SmolVLA base checkpoint. | models.smolvla_tmd.load_smolvla_tmd | yes | yes | training,inference,evaluation | lerobot/smolvla_libero |
| checkpoints.student_revision | str | "31d453f7edd78c839a8bbc39744a292686daf0de" | 40-character git commit | Immutable student model revision. | models.smolvla_tmd.load_smolvla_tmd | yes | yes | training,inference,evaluation | 31d453f7edd78c839a8bbc39744a292686daf0de |
| checkpoints.student_processor_revision | str | "31d453f7edd78c839a8bbc39744a292686daf0de" | 40-character git commit | Immutable student pre/postprocessor revision. | models.smolvla_tmd.load_smolvla_tmd | yes | yes | training,inference,evaluation | same as student_revision |
| checkpoints.teacher_id | str | "lerobot/pi05_libero_finetuned" | nonempty Hub ID or local path | Frozen pi0.5 teacher checkpoint; unused in B0-B2. | teacher.query.FrozenTeacherQuerier | yes | yes | training | lerobot/pi05_libero_finetuned |
| checkpoints.teacher_revision | str | "8e174154ef5f6c60a8da12ae99c303d8963138c1" | 40-character git commit | Immutable teacher model revision. | teacher.query.FrozenTeacherQuerier | yes | yes | training | 8e174154ef5f6c60a8da12ae99c303d8963138c1 |
| checkpoints.teacher_processor_revision | str | "8e174154ef5f6c60a8da12ae99c303d8963138c1" | 40-character git commit | Immutable teacher pre/postprocessor revision included in cache identity. | teacher.query.FrozenTeacherQuerier; teacher.cache.TeacherQueryCache | yes | yes | training,storage | same as teacher_revision |
| checkpoints.teacher_inference_steps | int | 10 | positive integer | Actual number of teacher denoising steps. | teacher.query.FrozenTeacherQuerier._apply_inference_steps | yes | yes | training | 10 |
| checkpoints.lerobot_commit | str | "3b2c0e59d557be5bd60ae4b1a6b51ba44936ebf6" | 40-character git commit | Required source revision for the local LeRobot dependency. | compatibility.lerobot_api.verify_lerobot_api | no | yes | training,inference,evaluation | 3b2c0e59d557be5bd60ae4b1a6b51ba44936ebf6 |
| dataset.task_suite | str | "libero_10" | libero_spatial, libero_object, libero_goal, libero_10, or libero_90 | LIBERO benchmark suite used for complete-episode evaluation. | evaluation.policy_runner.evaluate_policy_arm | yes | no | evaluation,storage | libero_10 |
| dataset.repo_id | str | "HuggingFaceVLA/libero" | nonempty Hub dataset ID | Expert dataset identity. | data.expert.load_lerobot_expert_dataset | yes | yes | training,evaluation,storage | HuggingFaceVLA/libero |
| dataset.revision | str | "86958911c0f959db2bbbdb107eb3e17c5f9c798e" | 40-character git commit | Immutable expert dataset revision. | data.expert.load_lerobot_expert_dataset | yes | yes | training,evaluation,storage | 86958911c0f959db2bbbdb107eb3e17c5f9c798e |
| dataset.split_seed | int | 17 | nonnegative integer | Seed for task-stratified episode splitting. | data.expert.episode_split_three_way | yes | no | training,evaluation,storage | 17 |
| dataset.validation_fraction | float | 0.1 | 0 <= value < 1 | Per-task fraction of complete episodes assigned to validation. | data.expert.episode_split_three_way | yes | no | training,evaluation,storage | 0.1 |
| dataset.test_fraction | float | 0.1 | 0 <= value < 1 and validation+test < 1 | Per-task fraction of complete episodes assigned to test. | data.expert.episode_split_three_way | yes | no | training,evaluation,storage | 0.1 |
| dataset.stride | int | 10 | positive integer | Frame stride between expert chunk starts. | data.expert.build_expert_chunks | yes | no | training,storage | 10 |
| dataset.episodes | tuple[int, Ellipsis] | [0, 1] | nonempty unique nonnegative integers | Complete expert episodes placed in scope. | data.expert.load_lerobot_expert_dataset | yes | no | training,evaluation,storage | [0, 1] for smoke; expand for experiments |
| tmd.outer_steps | int | 2 | positive integer | Number of outer Euler evaluations. | models.tmd.TMDActionGenerator | no | yes | training,inference,evaluation | 2 |
| tmd.inner_steps | int | 2 | positive integer | Transition-head Euler evaluations per outer step. | models.tmd.TMDActionGenerator | no | yes | training,inference,evaluation | 2 |
| tmd.inner_source_mode | str | "gaussian_tm" | gaussian_tm or anchored_tm_ablation; meanflow reserved | Distribution and parameterization of the inner transition flow. | models.transition_head.RecurrentTransitionHead | no | yes | training,inference,evaluation | gaussian_tm |
| tmd.recurrent_layers | int | 2 | positive integer | Number of recurrent GRUCell layers in the transition head. | models.transition_head.RecurrentTransitionHead | no | yes | training,inference | 2 |
| tmd.hidden_dim | int | 256 | even integer >= 2 | Transition-head recurrent width. | models.transition_head.RecurrentTransitionHead | no | yes | training,inference | 256 |
| tmd.dropout | float | 0.0 | 0 <= value < 1 | Transition-head inter-layer dropout probability. | models.transition_head.RecurrentTransitionHead | no | yes | training,inference | 0.0 |
| tmd.loss | str | "huber" | huber or mse | Elementwise transition-velocity regression loss. | models.transition_head.RecurrentTransitionHead.matching_loss | no | yes | training | huber |
| tmd.main_loss_weight | float | 0.0 | nonnegative float | Coefficient on optional direct backbone flow loss. | models.tmd.TMDActionGenerator.transition_matching_loss | no | yes | training | 0.0 |
| tmd.freeze_vision_encoder | bool | true | true only in the audited first experiment | Whether the large vision/language backbone remains frozen. | models.smolvla_tmd.SmolVLATMDPolicy.configure_trainable | no | yes | training | true |
| tmd.train_main_action_projections | bool | false | boolean | Whether small SmolVLA action/state/time projection modules are trainable. | models.smolvla_tmd.SmolVLATMDPolicy.configure_trainable | no | yes | training | false for B2 |
| discriminator.model_dim | int | 128 | positive and divisible by num_heads | Causal path-transformer embedding width. | models.discriminator.CausalPathDiscriminator | no | yes | training,evaluation | 128 |
| discriminator.num_layers | int | 3 | positive integer | Transformer encoder depth. | models.discriminator.CausalPathDiscriminator | no | yes | training,evaluation | 3 |
| discriminator.num_heads | int | 4 | positive divisor of model_dim | Self-attention head count. | models.discriminator.CausalPathDiscriminator | no | yes | training,evaluation | 4 |
| discriminator.feedforward_dim | int | 256 | positive integer | Transformer feed-forward width. | models.discriminator.CausalPathDiscriminator | no | yes | training,evaluation | 256 |
| discriminator.dropout | float | 0.1 | 0 <= value < 1 | Discriminator dropout probability. | models.discriminator.CausalPathDiscriminator | no | yes | training,evaluation | 0.1 |
| discriminator.num_tasks | int | 40 | positive integer | Size of the task embedding table. | models.discriminator.CausalPathDiscriminator | no | yes | training,evaluation | 40 |
| discriminator.weighting_min | float | 0.5 | positive and <= weighting_max | Minimum detached mismatch-emphasis weight. | training.distillation.MismatchPrioritization | no | yes | training | 0.5 |
| discriminator.weighting_max | float | 2.0 | >= weighting_min | Maximum detached mismatch-emphasis weight. | training.distillation.MismatchPrioritization | no | yes | training | 2.0 |
| discriminator.weighting_temperature | float | 1.0 | positive float | Temperature of mismatch-emphasis sigmoid. | training.distillation.MismatchPrioritization | no | yes | training | 1.0 |
| training.seed | int | 7 | nonnegative integer | Primary training RNG seed. | training.runner.seed_everything | no | yes | training | 7 |
| training.evaluation_seeds | tuple[int, Ellipsis] | [7, 17, 27] | nonempty unique nonnegative integers | Complete-episode evaluation reset seeds. | evaluation.policy_runner.evaluate_policy_arm | no | no | evaluation | [7, 17, 27] |
| training.task_indices | tuple[int, Ellipsis] | [0] | nonempty unique nonnegative integers | LIBERO task subset for the run. | evaluation.policy_runner.evaluate_policy_arm | yes | no | training,evaluation,storage | [0] for the first experiment |
| training.batch_size | int | 8 | positive integer | Training minibatch size. | training.runner.train_expert_chunk | no | no | training | 8 |
| training.expert_steps | int | 20 | nonnegative integer | Expert-only transition-head update count. | training.runner.train_expert_chunk | no | no | training | 20 smoke; increase for B2 |
| training.discriminator_steps | int | 50 | nonnegative integer | Discriminator optimizer update count. | cli._train_discriminator; experiments.motivation.runner.run_experiments | no | no | training | 50 smoke |
| training.distillation_steps | int | 20 | nonnegative integer; gated until B0-B2 | Teacher-distillation update count. | cli._gated | no | no | training | 0 until B0-B2 pass |
| training.learning_rate | float | 1e-05 | positive float | AdamW learning rate. | training.runner.make_optimizer | no | yes | training | 1e-5 for the first real-chunk experiment |
| training.weight_decay | float | 0.0001 | nonnegative float | AdamW decoupled weight decay. | training.runner.make_optimizer | no | yes | training | 1e-4 |
| training.expert_weight | float | 1.0 | nonnegative float | Coefficient on expert per-sample loss. | cli._gated; training.distillation.combined_distillation_loss | no | yes | training | 1.0 |
| training.teacher_weight | float | 1.0 | nonnegative float | Coefficient on teacher per-sample loss; gated until B3. | cli._gated; training.distillation.combined_distillation_loss | no | yes | training | 1.0 when B3 is enabled |
| training.replay_mode | str | "fresh_only" | fresh_only or historical_importance | Whether discriminator negatives are exact current samples or corrected replay. | cli._gated; training.replay.ReplayPools | no | yes | training,storage | fresh_only |
| training.minimum_fresh_current | int | 1 | positive integer | Minimum exact-current records before discriminator training. | cli._gated; training.replay.ReplayPools.require_fresh_current | no | no | training,storage | 1 smoke; increase for experiments |
| training.device | str | "cuda" | cpu, cuda, or cuda:N | Torch execution device. | models.smolvla_tmd.load_smolvla_tmd; training.runner.train_expert_chunk | no | no | training,inference,evaluation | cuda |
| evaluation.episode_length | int | 20 | positive integer | Explicit environment time limit for each complete first-experiment episode. | evaluation.policy_runner.evaluate_policy_arm | yes | no | evaluation,storage | 20 for plumbing feasibility only; use suite default for policy claims |
| evaluation.bootstrap_resamples | int | 1000 | positive integer | Number of task-stratified complete-episode bootstrap resamples. | evaluation.metrics.bootstrap_episode_statistic | no | no | evaluation | 1000 |
| evaluation.bootstrap_confidence | float | 0.95 | 0 < value < 1 | Percentile-bootstrap confidence level. | evaluation.metrics.bootstrap_episode_statistic | no | no | evaluation | 0.95 |
