# Experiment guide

Run from the repository root. Every command first prints the resolved config,
pinned revisions, canonical task mappings, output path, resource estimate, and
the exact real command. Dry-run never downloads data, loads weights, creates an
environment, queries a teacher, or trains.

## 1. Inspect and materialize data

```bash
# INSPECT ONLY
scripts/data/inspect_libero_tasks.sh --dry-run

# COLLECTS/MATERIALIZES EXPERT DATA (large; user-run only)
scripts/data/build_libero_expert.sh --dry-run
scripts/data/build_libero_expert.sh --execute

# INSPECT ONLY
scripts/data/audit_dataset.sh --dry-run
scripts/data/audit_dataset.sh --execute
```

The checked-in registry maps 90 cached LIBERO-10 episodes by exact instruction
and BDDL SHA-256. Any dataset task/instruction disagreement aborts before a
payload is written. Split assignment is per episode and stratified by canonical
task UID.

## 2. Train Flow-SFT

```bash
# TRAINS when --execute is used
scripts/train/train_flow_sft.sh --dry-run
scripts/train/train_flow_sft.sh --execute
# Explicit deterministic resume:
scripts/train/train_flow_sft.sh --execute --resume artifacts/research/flow_sft/checkpoint.pt
```

The executable adapter uses the official pinned SmolVLA processor, Cond-OT
loss, normalization, masks, mixed precision, accumulation, full multi-record
DataLoader, deterministic per-epoch shuffle, fixed validation noise/time, and a
format-v3 complete checkpoint.

## 3–7. TMD, DMD2-flow, and OPD

```bash
# TRAINS Stage 1 after its end-to-end adapter is completed
scripts/train/train_tmd_stage1.sh --dry-run
# FAILS CLOSED: pinned pi0.5 lacks score-at-student-action
scripts/train/train_tmd_stage2.sh --dry-run
scripts/train/train_dmd2_flow.sh --dry-run

# COLLECTS current-student environment trajectories when adapter is completed
scripts/data/collect_student_rollouts.sh --dry-run
# QUERIES frozen teacher, never executes its actions
scripts/data/query_pi05_teacher.sh --dry-run
# FAILS CLOSED for continuous pi0.5 density; categorical config is separate
scripts/train/train_opd_on_policy.sh --dry-run
```

The equation/loss/model/checkpoint components are implemented, but the generic
launcher will record a failed attempt instead of pretending that an unavailable
pi0.5 density/score or an unwritten environment adapter exists. No velocity-MSE
surrogate is labeled OPD.

The mandatory legacy comparison is explicitly named and never reported as
paper TMD:

```bash
scripts/train/train_tmd_plain_gaussian_ablation.sh --dry-run
```

## 8–9. Occupancy discriminator and occupancy TMD

```bash
# TRAINS only after matched real expert/current-policy stores and adapter exist
scripts/train/train_occupancy_discriminator.sh --dry-run
# FAILS CLOSED until the real held-out gate passes and TMD Stage 2 is available
scripts/train/train_occupancy_tmd.sh --dry-run
```

Synthetic AUC cannot open the gate. Required diagnostics include calibration,
source-only and balanced controls, support overlap, saturation, ESS, freshness,
and clipping.

## 10. Evaluate and compare

```bash
# EVALUATES only; all methods must use this shared protocol
scripts/evaluate/evaluate_policy.sh --dry-run
scripts/evaluate/compare_methods.sh --dry-run
scripts/evaluate/evaluate_discriminator.sh --dry-run
scripts/evaluate/measure_latency.sh --dry-run
```

Local and Slurm preflight templates are `scripts/launch/launch_local.sh` and
`scripts/launch/launch_slurm.sh`. Never submit a command until its dry-run says
`executable: true`, the task mappings are correct, and the output path is new
or an explicit resume checkpoint is supplied.
