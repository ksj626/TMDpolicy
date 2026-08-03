# Final implementation report

## 1. Executive conclusion

The repository is now an audited, modular research codebase whose primary inner
source is verified standard Gaussian noise. The old backbone-anchored method is
available only as the explicit `anchored_tm_ablation`; reserved MeanFlow fails
closed. Mathematical, configuration, checkpoint, teacher, schema, storage,
rollout, discriminator, replay-semantic, documentation, and motivation layers
are implemented and covered by 59 tests.

The code was also exercised with the pinned real SmolVLA checkpoint and in real
LIBERO simulation. B0/B1/B2 each completed three task-0, 20-step time-limited
episodes at seeds 7/17/27 with schema-v2 provenance. All arms had zero success at
that deliberately short horizon. The coarse arms reduced warm model latency per
replan from 279.7 ms to 101.5 ms (`~2.754x`). This establishes feasibility and
latency, not policy quality. B3/B4 remain gated.

## 2. Audit findings by severity

The full line-level table is in `docs/audit_report.md`.

- P0: inner source/target parameterization used backbone anchor rather than
  Gaussian `Z`; the trained target belonged to the anchored construction.
- P1: whole-batch loss reduction, incomplete checkpoint state, silent config
  keys, incomplete teacher identity/runtime steps, permissive schemas, unlocked
  storage, and missing processor provenance.
- P2: untracked inner randomness, assumed KV integrity, absent discriminator
  normalization/balancing, conflated replay/mismatch terminology, incomplete
  rollout timing, no API pin, and no episode bootstrap.
- P3/verified: the outer `t:1->0` sign and oracle endpoint were already correct.

An additional real-run finding was that the pinned LeRobot LIBERO wrapper stores
an episode length but its `step()` always returns `truncated=False`. The project
collector now enforces/records its own configured local time limit; LeRobot was
not modified.

## 3. Mathematical correction

Old explicit ablation (formerly implicit/default):

```text
source = B
Y_tau = (1-tau)Y + tau B
dY_tau/dtau = B-Y
current starts at B; network directly predicts the inner velocity
```

New default:

```text
A_t=(1-t)A+t epsilon,       Y=epsilon-A
Z~N(0,I), independent,      Y_s=(1-s)Y+sZ
dY_s/ds=Z-Y,                Y_hat=B+Delta_theta
u_theta=Z-Y_hat,            ds=-1/inner_steps, s:1->0
```

Zero `Delta` returns `B` exactly; oracle `Delta=Y-B` returns `Y`. Losses first
reduce valid horizon/action coordinates per sample. Fixed outer plus all inner
sources reproduce sampling. Only the head and explicitly enabled small
projections receive gradients.

## 4. Files changed

The starting project was not a Git worktree, so scope is listed by file group.
All writes are under `/home/dmsdmswns/TMDpolicy`.

- Core/config/CLI: `src/tmd_policy/{config.py,cli.py,smoke.py,README.md}` and both
  YAML configs.
- Models: all files in `src/tmd_policy/models/` plus its README.
- Data: `schemas.py`, `storage.py`, `expert.py`, and README.
- Rollout/teacher/compatibility/evaluation: collectors, query/cache, new
  `lerobot_api.py`, metrics, new `policy_runner.py`, and READMEs.
- Training: checkpoint, runner, diagnostics, discriminator, distillation,
  replay, alternating orchestration, and README.
- Experiments: new `experiments/motivation/` package, config, runner, synthetic
  data, plotting, CLI, and READMEs.
- Scripts/tests/docs: generated-reference script; expanded tests; root and all
  required module READMEs; audit, method, contract, milestone, config reference,
  and this report.
- Artifacts: additive milestone 0/1/2/3/5/6 and final-validation evidence. No
  pre-existing artifact was overwritten.

## 5. Documentation created

Required root/config/scripts/tests/experiments/module READMEs exist and are
tested for per-file coverage. They contain file/API/caller/consumer/input/output,
trainable/side-effect/config/artifact/limitation tables; tensor and scalar
dictionaries; and eleven Mermaid flows covering the nine required systems.
`docs/config_reference.md` is generated from the actual dataclass metadata and
checked byte-for-byte.

## 6. Tests added

New regressions cover Gaussian statistics/path/velocity/endpoints, zero/oracle
residuals, CPU/CUDA, masks/per-sample reductions, gradients, noise determinism,
explicit/reserved modes, config keys/consumers/docs, adversarial schemas,
locking/corruption/recovery, cache collisions/inference steps, deterministic
resume, train-only normalization, typed weights, episode bootstrap, seeded
rollout replay/local truncation, KV mutation, episode-disjoint motivation data,
plot reproducibility, README/diagram/file coverage, and CLI completeness.

## 7. Exact commands run

Representative successful commands (all from project root):

```bash
PYTHONPATH=src:. conda run --no-capture-output -n lerobot pytest -q
PYTHONPATH=src:. conda run -n lerobot ruff check .
PYTHONPATH=src:. conda run --no-capture-output -n lerobot python -m compileall -q src experiments tests
PYTHONPATH=src:. conda run -n lerobot python scripts/generate_config_reference.py --check

PYTHONPATH=src:. conda run --no-capture-output -n lerobot python experiments/motivation/run.py --experiments M0 --output artifacts/milestone5_20260803/motivation --seed 101
PYTHONPATH=src:. conda run --no-capture-output -n lerobot python experiments/motivation/run.py --experiments M1 M2 M3 M4 M5 --output artifacts/milestone5_20260803/motivation --seed 101

CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src:. conda run --no-capture-output -n lerobot python -m tmd_policy.cli train-expert --config configs/tiny.yaml --expert-manifest artifacts/expert_real/manifest.jsonl --output artifacts/milestone6_20260803/B2_training_v2 --record-index 0

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. MUJOCO_GL=egl conda run --no-capture-output -n lerobot python -m tmd_policy.cli evaluate-policy --config configs/tiny.yaml --arm B0 --output artifacts/milestone6_20260803/B0_evaluation_v5
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. MUJOCO_GL=egl conda run --no-capture-output -n lerobot python -m tmd_policy.cli evaluate-policy --config configs/tiny.yaml --arm B1 --output artifacts/milestone6_20260803/B1_evaluation
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src:. MUJOCO_GL=egl conda run --no-capture-output -n lerobot python -m tmd_policy.cli evaluate-policy --config configs/tiny.yaml --arm B2 --checkpoint artifacts/milestone6_20260803/B2_training_v2/checkpoint.pt --output artifacts/milestone6_20260803/B2_evaluation

PYTHONPATH=src:. conda run --no-capture-output -n lerobot python -m tmd_policy.cli audit --config configs/tiny.yaml --output artifacts/final_validation_20260803/audit_v2 --store artifacts/milestone6_20260803/B0_evaluation_v5/rollout_chunks --store artifacts/milestone6_20260803/B1_evaluation/rollout_chunks --store artifacts/milestone6_20260803/B2_evaluation/rollout_chunks
```

The environment variables `HF_HOME` and `HF_LEROBOT_HOME` were also pointed to
the project cache for real commands. Failed/interrupted exact attempts and their
reasons are preserved in `artifacts/milestone6_20260803/attempts.md`, including
the GPU-4 EGL traceback and the non-enforced LeRobot time limit.

## 8. Test results

- Final pytest: 59 passed in 15.44 s.
- Ruff: all checks passed.
- Compileall: exit 0.
- Generated config reference check: exit 0.
- LeRobot commit/API audit: compatible at the pinned commit.
- B0/B1/B2 schema-v2 store audits: no manifest, missing, orphan, or temporary
  payload issue.
- Format-v2 interrupted/resumed update: exact parameter/loss/RNG equality.

Machine-readable validation and hashes are in
`artifacts/final_validation_20260803/`.

## 9. Motivation experiments

M0 passed: expert-A/expert-B AUC 0.4872, current-A/current-B 0.4862, and
expert/current 0.8306; maximum control deviation from chance was 0.0138. M1
pointwise/final/prefix AUC was 0.838/0.862/0.848. M2 synthetic score-success AUC
was 0.996 with episode-bootstrap success-minus-failure logit 3.459 (95% CI
3.314–3.629). M3 failure/increment correlation was 0.281. M4 is explicitly a
synthetic perturbation. M5 is explicitly a synthetic coarse proxy and records
latency as unavailable rather than fabricating it.

## 10. Figures and raw metrics

All M0–M5 outputs are below `artifacts/milestone5_20260803/motivation/M*/`.
Each directory has `metrics.json`, NPZ/CSV raw inputs, and PNG+SVG figures. The
M0 gate and aggregate run list are `M0/metrics.json` and `run_summary.json`.

## 11. Real versus synthetic evidence

- Implemented: all audited modules and CLI surface; B3/B4 commands fail closed.
- Tested synthetically: full math/system suite and M0–M5 motivation diagnostics.
- Tested with real expert data: canonical action-chunk diagnostic reduced loss
  3735x using a diagnostic backbone; clearly labeled as such.
- Tested with the real checkpoint: B2 trained 20 updates using actual SmolVLA;
  fixed diagnostic loss fell `0.005178→0.005019`. Its stochastic last minibatch
  loss was high (`0.1509`) and is not hidden or interpreted as convergence.
- Tested in LIBERO: B0/B1/B2, task 0, three seeds, six chunks/arm, 20-step local
  limit, zero successes; real warm latency and action differences saved.
- Not tested: suite-default-horizon success, real discriminator motivation,
  pi0.5 queries, B3/B4 distillation, or replay training.

## 12. Remaining blockers

The quality gate remains closed because the real evaluation horizon was only 20
steps and all arms failed. B2 used one expert observation and 20 updates, not a
representative episode/task training split. Real M0 controls require enough
fresh B0/B1/B2 episodes on matched tasks. Teacher queries/replay must wait for
that evidence. The pinned dependency's missing time-limit truncation is handled
locally but should be reported upstream separately, without patching this setup.

## 13. Next smallest experiment

Build schema-v2 expert data from at least three complete task-0 episodes split by
episode, train B2 on all train chunks with a held-out fixed diagnostic, then run
B0/B1/B2 at the suite-default task-0 horizon for seeds 7/17/27. Compare success,
warm per-replan latency, and held-out action/TM metrics. If that passes, collect
enough exact-current B2 episodes to rerun M0/M1 on real matched sources. Only
then open one unweighted B3 teacher-query experiment; keep B4/replay gated.
