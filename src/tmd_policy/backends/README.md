# Backend contract

`protocols.py` defines the narrow flow-policy interface used by algorithms.
`action_coordinates.py` owns differentiable canonical/normalized action maps.
`lerobot/compatibility.py` pins LeRobot 0.6.1 APIs and hashes imported sources.
`lerobot/pi05_teacher.py` owns the frozen PI0.5 processor, prefix KV cache,
one-step velocity, score conversion, and deterministic Euler sampler.
`lerobot/smolvla_student.py` owns the SmolVLA processor, official Flow-SFT
objective, explicit trainable-module modes, and shared sampler.

Actions use `[B,50,7]` canonical LIBERO coordinates or `[B,50,32]` checkpoint
coordinates. `True` valid masks mean an environment action exists. PI0.5 is
always frozen; student gradients pass through its action-coordinate bridge but
not through teacher outputs. Prefix caches are observation-local GPU objects and
are never written into datasets or checkpoints.

Public API: `CanonicalBatch`, `FlowCondition`, and `FlowPolicy` define the
algorithm-facing protocol; `ActionNormalizer` implements official affine
normalization; `ActionCoordinateBridge` returns `CoordinateResult` values and
masks. The LeRobot subpackage owns the concrete teacher/student caches and
adapters described in its README.
