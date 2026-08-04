# LeRobot adapters

These files consume the installed PyPI package `lerobot==0.6.1`; they never
patch it. Every run records paths and SHA-256 hashes for PI0.5, SmolVLA,
normalization, and processor-factory sources. Model and processor revisions are
immutable Hub commits. A mismatch fails before data collection or training.

PI0.5 follows `x_t=(1-t)a+tε`, velocity `ε-a`, and score
`-(x_t+(1-t)v)/t`. Score time is visibly clamped by `minimum_score_time`.
SmolVLA uses the same descending Euler convention. Both operate in float32
action tensors on their configured devices; model activations may be BF16.

`verify_installed_lerobot` and `LeRobotCompatibilityError` are the fail-closed
compatibility API. `LeRobotPI05Teacher` creates `PI05ConditionCache` objects;
`cache_fingerprint` verifies cache immutability. `LeRobotSmolVLAStudent` creates
`SmolVLAConditionCache` objects and owns explicit `FineTuningMode` selection.
Model sampling noise is caller-owned; processor tokenization/image transforms
follow the pinned checkpoint. Checkpoints store identities, never live caches.
