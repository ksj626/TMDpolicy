# LIBERO evaluation

`policy.py` reconstructs only the student/sampler needed for inference and
verifies checkpoint SHA, method, model, processor, and dataset identities.
`libero.py` runs complete real episodes with deterministic fixed-noise replans,
official environment/policy processing, synchronized latency, task success,
Wilson intervals, action-path smoothness, and optional versioned rollout
payloads. `compare.py` requires identical `(suite, task_id, reset_seed)` grids
and reports overall, per-suite, and per-task paired bootstrap success
differences plus exact McNemar tests.

Evaluation actions are officially postprocessed canonical `[50,7]` chunks;
only the configured execution horizon is stepped before replanning. Main and
motivation configs deliberately cover multiple suites and many paired seeds to
make method differences statistically visible, at correspondingly high compute
cost. No synthetic episode is accepted by these paths.

`InferencePolicy`/`load_inference_policy` reconstruct the shared sampler;
`run_episode` executes one seed; `evaluate_libero` and
`collect_student_rollouts` run configured grids; `summarize` and
`wilson_interval` produce single-arm metrics; `compare` performs paired-arm
statistics. Reset seeds and derived policy-noise seeds are deterministic; model
and environment latency are synchronized when configured.
