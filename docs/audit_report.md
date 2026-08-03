# Mathematical and systems audit

Audit date: 2026-08-03. Line references in the findings table are **pre-fix
line numbers** from the Milestone 0 snapshot. The immutable baseline inventory,
hashes, environment, and test output are in `artifacts/milestone0_20260803/`;
the expected red test is in `artifacts/milestone1_20260803/`.

## Audited method contract

For canonical actions `A ∈ R^{B×50×7}`, outer Gaussian noise `ε` has exactly
the same shape, dtype, and device. The outer path and constant target velocity
are

```text
A_t = (1 - t) A + t ε,       Y = dA_t/dt = ε - A.
```

At every outer Euler evaluation, the frozen/main backbone predicts a reference
`B = B(A_t, t, context)` and features `F`. The default transition-matching
method draws a new `Z ~ N(0, I)` with `Z.shape == Y.shape` and uses

```text
Y_s = (1 - s) Y + s Z,       dY_s/ds = Z - Y,       s: 1 -> 0.
Y_hat = B + Delta(Y_s, Z, B, A_t, t, s, F)
u_theta = Z - Y_hat.
```

Euler integration uses `ds = -1 / inner_steps`. Therefore a zero residual gives
`u_theta = Z - B` and returns `B` exactly, while an oracle residual
`Delta = Y - B` gives `u_theta = Z - Y` and returns `Y` exactly. The explicit
`anchored_tm_ablation` mode retains the historical backbone-anchored path only
for controlled comparison. `gaussian_tm_meanflow` is a reserved, rejected mode;
it must never silently alias either implemented method.

## Findings and disposition

| ID | Severity | Pre-fix evidence | Observed behavior | Intended behavior | Category | Correction and regression test | Status |
|---|---|---|---|---|---|---|---|
| MATH-01 | P0 | `models/transition_head.py:24-27,107,137-145` | The inner path starts at the backbone transition and trains a direct velocity from that anchor. This is not Gaussian transition matching. | Independent `Z ~ N(0,I)`, analytic `Y_s`, residual reference `Y_hat=B+Delta`, and negative-time integration. | mathematical correctness | Add explicit source modes; make `gaussian_tm` default; test endpoints, formula, zero residual, and oracle residual on CPU/CUDA. | fixed |
| MATH-02 | P0 | `models/transition_head.py:138,155-157` | The trained target is `B-Y`, tied to the anchored construction. | Target is `Z-Y`; predicted velocity is `Z-(B+Delta)`. | mathematical correctness | Compute the velocity loss literally and test its algebra. | fixed |
| MATH-03 | P1 | `models/transition_head.py:162-165`; `models/tmd.py:107-114` | Masked errors collapse the entire batch to one scalar before distillation weighting. | First reduce valid positions and action coordinates per sample, then optionally reduce the batch. | loss semantics | Add `reduction='none'|'mean'` throughout and a heterogeneous-mask test. | fixed |
| MATH-04 | P2 | `models/tmd.py:57-77` | Sampling exposes only outer noise; inner stochastic sources cannot be reproduced independently. | One explicitly tracked Gaussian inner source per outer evaluation. | reproducibility | Accept `[outer_steps,B,H,D]` inner noises or a generator; persist outer/inner seeds. | fixed |
| MATH-05 | P3 | `models/tmd.py:23-32,60,64,73` | Outer convention is implemented consistently with integration from 1 to 0. | Preserve the convention and document every tensor/time direction. | mathematical correctness | Keep oracle sign/endpoint tests. | verified |
| TRAIN-01 | P1 | `models/smolvla_tmd.py:122-134,162-180` | Freezing is explicit, but checkpoints save only the transition head and omit optimizer, scheduler, scaler, RNG, mode, dimensions, trainable projections, and base identities. | Full deterministic resume state plus architecture/method metadata. | training/reproducibility | Central checkpoint module and uninterrupted-vs-resumed test. | fixed |
| TRAIN-02 | P2 | `models/smolvla_tmd.py:31-78` | Prefix KV caching is used, but cache immutability across inner/outer calls is assumed rather than checked. | Reuse one prefix cache and fail if suffix calls mutate its structure. | performance/correctness | Cache-signature guard and fake-cache regression test. | fixed |
| TRAIN-03 | P2 | `models/discriminator.py:18-88`; `training/discriminator.py:10-35` | Causal masking and expert-positive orientation are correct, but no train-split normalizer or task/position balancing exists. | Fit normalization on the training split only and balance task/position contributions. | discriminator design | Registered path normalizer, balanced loss/sampler utilities, causal and no-leakage tests. | fixed |
| TRAIN-04 | P2 | `training/distillation.py:9-42`; `training/replay.py:8-24` | Detached bounded mismatch weighting is sound, but generic `weights()` naming makes replay importance and failure emphasis easy to conflate. | Distinct types and names for mismatch emphasis and importance correction; report ESS. | objective semantics | Rename APIs/types and retain compatibility alias with warning-free explicit semantics. | fixed |
| DATA-01 | P1 | `data/schemas.py:47-51,109-112,162-163` | Schemas accept arbitrary horizons and dimensions and do not reject non-finite values or non-prefix masks. | Expert/teacher plan `(50,7)`, expert path `(10,7)/(11,8)`, rollout plan `(50,7)`, executed prefix `0..10`, exact state relation, finite values, contiguous masks. | schema validation | Versioned strict schemas and adversarial shape/mask tests. | fixed |
| DATA-02 | P1 | `data/storage.py:27-50` | Duplicate check, payload rename, and manifest append are not protected by a writer lock; crashes may orphan payloads or leave a partial JSON line. | Single writer, atomic payload publication, fsynced manifest, detectable/recoverable partial/orphaned records. | storage integrity | Exclusive lock and read-only audit plus explicit recovery API; multiprocessing and corruption tests. | fixed |
| DATA-03 | P1 | `teacher/cache.py:14-34`; `data/schemas.py:166-185`; `teacher/query.py:45-75` | Cache identity omits processor revision; sample digest also omits inference steps. | Key includes observation, teacher checkpoint/revision, processor revision, inference steps, sampling seed, and sample index. | provenance | Canonical hashed key and collision tests. | fixed |
| DATA-04 | P1 | `teacher/query.py:25-33,54-59` | `teacher_inference_steps` is recorded but never applied to the teacher policy. | Set and verify the runtime inference-step field before querying. | configuration correctness | Runtime setter/check and fake-teacher test. | fixed |
| DATA-05 | P2 | `rollout/collector.py:36-54,68-83,136,141-145` | Only aggregate policy latency and one seed are recorded; CUDA synchronization and preprocessing/model/postprocessing segments are absent. | Synchronized segmented timing and separate outer/inner seed provenance per chunk. | rollout instrumentation | Structured `PlanResult` and schema fields with deterministic tests. | fixed |
| API-01 | P2 | `models/smolvla_tmd.py:31-78,183-217` | LeRobot internals are called directly without a startup signature/attribute check or source commit assertion. | Fail fast against the pinned LeRobot API/commit with an actionable compatibility report. | dependency compatibility | Add `compatibility/lerobot_api.py`, tests, and CLI audit. | fixed |
| CFG-01 | P1 | `config.py:94-106` | Unknown keys are silently ignored. Several declared values have no executable consumer. | Reject unknown values recursively and produce a deterministic field-to-consumer report/reference. | configuration | Strict loader, validators, consumption registry, generated reference, CI test. | fixed |
| EVAL-01 | P2 | `evaluation/metrics.py:1-132` | Position/task/failure diagnostics exist but episode bootstrap confidence intervals do not. | Report episode-level bootstrap CIs for final comparisons. | evaluation | Add deterministic stratified episode bootstrap utility and tests. | fixed |
| DOC-01 | P2 | root `README.md`; `docs/method_report.md`; package directories | Prototype documentation lacks a repository-wide map, tensor dictionaries, public API contracts, executable command matrix, and motivation experiment interpretation. | Documentation is part of the research artifact and must match executable code. | documentation | Rewrite root/module READMEs, generated config reference, diagrams, experiment READMEs, and doc-consistency tests. | fixed |

## Line-by-line training mapping

The implementation in `models/tmd.py` and `models/transition_head.py` follows
this sequence:

1. Validate action, mask, outer-noise, inner-noise, and time tensors.
2. Sample (or accept) `ε`; form `A_t` and `Y=ε-A`.
3. Evaluate the backbone exactly once at this outer sample to get `(B,F)`.
4. Sample (or accept) a shape/device/dtype-matched `Z`.
5. For each descending inner time `s`, construct `Y_s` analytically.
6. Carry recurrent hidden state between inner evaluations.
7. Predict `Delta`, form `Y_hat=B+Delta`, and then `u_theta=Z-Y_hat`.
8. Compare `u_theta` with `Z-Y` elementwise.
9. Reduce over valid horizon positions and action coordinates **per sample**.
10. Average inner evaluations; only `reduction='mean'` averages the batch.
11. Compute the optional main-flow error using the identical per-sample rule.
12. Return named per-sample components so later mismatch weighting is applied
    before any batch reduction.

## Sampling mapping

Sampling creates one outer `ε`, builds one language/vision prefix KV cache, and
per outer step performs exactly one backbone call followed by `inner_steps`
lightweight transition-head calls. A distinct `Z` is used for each outer step.
The head integrates from `s=1` to `s=0`; the resulting transition velocity is
then used by the outer Euler step from `t=1` to `t=0`. Neither API accepts target
actions at inference time.

