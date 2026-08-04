# Research-grade refactor completion report

Date: 2026-08-04. No expensive collection, download, teacher query, policy
evaluation, or training was run.

## Implemented

- Mathematical contracts for Flow-SFT, action-space TMD, DMD2-flow,
  categorical/continuous OPD, and proposed occupancy TMD.
- Shared canonical task registry, schema-v3 hashed store, strict images/masks,
  real multi-record DataLoader, episode splits, processors/actions, exact and
  Hutchinson density, outcome/metric definitions, complete checkpoints, and
  dirty dependency provenance.
- Isolated method classes, objectives, models, optimizer schedules, capability
  gates, sampling, and checkpoint methods.
- Real expert materializer and real official Flow-SFT training adapter.
- All requested shell entry points, method configs, local/Slurm templates, and
  dry-run preflight. Failed executable adapters preserve a failure artifact.
- Legacy repairs: `tmd.loss`/`dropout` propagation, multi-record training,
  success accumulation, distinct ending flags, AP versus PR-AUC, image/mask
  validation, and new payload hashes.

## Deliberate fail-closed status

- TMD Stage 2, DMD2-flow, and occupancy TMD cannot run against pinned pi0.5:
  its supported public LeRobot API has sampling and an internal denoise step,
  but no supported score/log-density at a student action.
- Continuous-flow OPD cannot run for the same reason. The exact categorical
  implementation is usable only with a provider of normalized token logits.
- TMD Stage 1, rollout/teacher adapters, occupancy training adapter, and the new
  shared evaluator still require end-to-end model/environment wiring. Their
  mathematical components exist, but they are not reported complete merely
  because a CLI exists.
- TMD paper Eq. 13 has an apparent sign inconsistency. The port explicitly uses
  the endpoint-consistent subtraction and is labeled an action-flow adaptation
  pending author code/erratum.

## Validation policy

Only tiny deterministic CPU tests, static checks, dry-runs, and the pre-existing
test suite are permitted. Exact commands and results are updated at final
handoff; real checkpoints/environments are deliberately not exercised here.

## Validation actually run

- `conda run -n lerobot env PYTHONPATH=src:. pytest -q` — **79 passed**;
  18 upstream Torch deprecation warnings, no failures.
- `conda run -n lerobot ruff check src tests` — passed.
- `bash -n scripts/data/*.sh scripts/train/*.sh scripts/evaluate/*.sh
  scripts/launch/*.sh` — passed.
- `git diff --check` and imports of the research CLI/method package — passed.
- Flow-SFT and task-inspection dry-runs — executable with 90 exact registered
  LIBERO-10 episodes. DMD2/TMD Stage 2 dry-runs correctly reported missing
  `flow_score` and `teacher_at_student_action`.

The repository base is `e426dd8646cd2b3fd4db901ced3f8f6f3a5fe0ff`.
The read-only LeRobot dependency is at
`3b2c0e59d557be5bd60ae4b1a6b51ba44936ebf6`, dirty, with complete tracked and
untracked patch hash
`05faff2fd95aa9600ce275db4280e60a98ba27d70c35a452ee3af36fa3e284c3`.
Real runs save that exact patch rather than trusting the commit alone.
