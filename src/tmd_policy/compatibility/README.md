# Compatibility

| File | Purpose | Public API | Caller → downstream | Input → output | Trainable parameters | Side effects / config / artifacts | Limitations |
|---|---|---|---|---|---|---|---|
| `actions.py` | Canonical action bounds, official processor bridge, explicit 8-D student/teacher state adapter. | `CanonicalActionSpace`, `PolicyActionBridge`, `StateCompatibilityAdapter` | rollout/teacher/tests → policy/env | canonical tensors ↔ policy tensors | none | processor calls only; canonical config | Fixed LIBERO 8-state/7-action contract. |
| `metadata.py` | Compare pinned dataset/student/teacher Hub metadata. | `CompatibilityReport`, `inspect_compatibility` | `inspect` CLI → JSON evidence | URLs and immutable IDs → report | none | reads local/HTTPS JSON; writes via caller | Metadata cannot prove runtime equivalence alone. |
| `lerobot_api.py` | Pin local LeRobot commit and consumed SmolVLA signatures/attributes. | `LeRobotCompatibilityError`, `verify_lerobot_api` | loader/audit → fail-fast report | expected commit and optional policy → compatibility map | none | read-only git/API inspection | Deliberately tied to audited internal API. |
| `__init__.py` | Stable canonical exports. | action bridge exports | package callers | n/a | none | none | no behavior |

| Variable | Type/shape | Coordinates/range | Phase/gradients | Producer → consumer / provenance |
|---|---|---|---|---|
| canonical action | `[...,7]` float tensor | LIBERO `[-1,1]`, postprocessed | train/infer/eval; gradient preserved only across tensor operations | policy-specific postprocessor → env/store/cross-policy loss |
| canonical state | `[...,8]` float tensor | LIBERO low-dimensional state, finite | train/infer; no environment gradient | dataset/env processor → student/teacher adapter |
| processor round-trip error | nonnegative float | canonical action units | audit only | official pre+postprocessor → compatibility report |
| expected/installed commit | 40-char strings | git identity | startup lifetime | strict config/git → loader gate/checkpoint provenance |

The student config historically declares six state dimensions while its pinned
normalizer consumes eight. The adapter therefore preserves the canonical 8-D
state and never truncates it implicitly. Processor revisions are independent
cache/checkpoint identities.
