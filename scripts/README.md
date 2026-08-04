# Executable scripts

`setup/create_environment.sh` creates only the fixed environment and verifies
it. `data/build_libero_expert.sh`, `data/query_pi05_teacher.sh`, and
`data/collect_student_rollouts.sh` build the real split manifest, run real PI0.5
parity, and collect real student LIBERO paths. Every file under `train/` invokes
one concrete trainer. `evaluate/evaluate_policy.sh` runs the expanded motivation
grid; `evaluate/compare_methods.sh` performs paired statistics. Arguments after
the script name are forwarded to the CLI, including `--output` and `--resume`.

Scripts set headless EGL and repo-local Hugging Face/LeRobot caches. They assume
the `tmdpolicy` Conda environment and never start from setup automatically.
