# Milestones and executed evidence

No number below is claimed without a saved artifact from an executed command.
Pre-existing prototype artifacts were hashed before any change and not
overwritten.

| Milestone | Status | Evidence |
|---|---|---|
| 0 — reproduce | complete | `artifacts/milestone0_20260803/`: environment, LeRobot commit/dirty-state record, artifact inventory/SHA-256, compile pass, 22 baseline tests pass. |
| 1 — audit | complete | `docs/audit_report.md`; saved expected red test in `artifacts/milestone1_20260803/`. |
| 2 — Gaussian TM | complete | endpoint/source/mask/gradient/determinism tests; synthetic report and real-expert-action diagnostic in `artifacts/milestone2_20260803/`. |
| 3 — systems | complete in code/tests | strict config/docs, format-v2 checkpoint resume, cache identity/runtime steps, schema-v2/locking/recovery, LeRobot API pin, segmented rollout seeds/timing. |
| 4 — documentation | complete | root + all required module READMEs, generated config reference, nine Mermaid flows, tensor/scalar dictionaries. |
| 5 — motivation | complete as synthetic diagnostics | M0–M5 raw/metrics/PNG/SVG in `artifacts/milestone5_20260803/motivation/`; explicitly not robot evidence. |
| 6 — minimal policy experiment | complete as bounded feasibility; quality gate remains closed | Real task 0, seeds 7/17/27, explicit 20-step local truncation, schema-v2 chunks, B2 checkpoint, and comparison in `artifacts/milestone6_20260803/`. |

## Key executed diagnostic results

| Check | Result |
|---|---:|
| Final audited tests | 59 passed |
| Synthetic Gaussian TM loss | `0.0264516 → 0.000190986` |
| Real LIBERO expert action-chunk diagnostic | `0.00655412 → 1.7548e-6` (`3734.96×`) |
| M0 expert-A vs expert-B ROC-AUC | `0.48716` |
| M0 current-A vs current-B ROC-AUC | `0.48620` |
| M0 expert vs current ROC-AUC | `0.83057` |
| M1 pointwise / final / prefix AUC | `0.838 / 0.862 / 0.848` |
| M2 score-as-success-predictor AUC | `0.996` synthetic; discriminator not trained on success |
| M2 success-minus-failure logit, 95% episode bootstrap | `3.459`, CI `[3.314,3.629]` |
| M3 negative-increment/failure correlation | `0.281` synthetic |
| M4 success standard/perturbed | `0.451/0.000` synthetic perturbation only |
| M5 original/coarse success proxy | `0.453/0.000` synthetic; latency intentionally unavailable |
| Real B0/B1/B2 feasibility success | `0/3`, `0/3`, `0/3` within 20 steps |
| Real warm model latency per replan B0/B1/B2 | `279.7 / 101.5 / 101.5 ms` |
| Real warm coarse-arm speedup | B1 `2.754×`, B2 `2.754×` vs B0 |
| Real first-plan B1-vs-B2 canonical MAE | `0.001054` across three seeds |
| B2 fixed real-chunk diagnostic loss | `0.005178 → 0.005019` over 20 updates |

M0 control deviation is at most `0.0138`, well under its `0.12` stop threshold,
so later synthetic diagnostics were permitted. M1 shows all three variants detect
the constructed shift; prefix value is supported by M3 temporal localization,
not by a claim that prefix has the highest overall AUC. M4/M5 are motivation
pipeline proxies and must not be presented as LIBERO robot results.

The earlier real smoke artifacts retain an official action round-trip error
`5.96e-8`, one untrained plan/10-action/11-state rollout, and prototype latency,
but they predate schema v2 and the corrected Gaussian source. They are baseline
evidence only, not B0/B1/B2 final comparisons.

The pinned LeRobot LIBERO wrapper stores an episode length but returns
`truncated=False` unconditionally. A failed attempt and traceback are preserved
in the milestone directory; the project collector now enforces and records the
explicit 20-step local limit without changing LeRobot. These are complete
time-limited feasibility episodes, not full suite-horizon quality evaluations.
Because no arm succeeded, the B3/B4 teacher-quality gate remains closed.
