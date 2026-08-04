# Python package map

`config.py` loads and validates immutable YAML contracts. `cli.py` defines real
data, teacher, train, rollout, and evaluation verbs. `backends/` isolates pinned
LeRobot internals and action coordinates. `data/` owns the single expert schema
and occupancy windows. `methods/` owns Flow-SFT, TMD/Stage-2, DMD2, and occupancy
program graphs. `training/` owns deterministic DataLoaders, optimizers, and full
resume. `rollout/` owns versioned student episodes. `evaluation/` owns policy
loading, complete LIBERO episodes, and paired statistics. `integration/` owns
the real fixed-noise PI0.5 parity check.

Public tensors are torch tensors. Canonical LIBERO actions are `[B,50,7]`;
checkpoint flow actions are `[B,50,32]`; valid masks are boolean `[B,50]` with
`True` meaning an environment target exists. Each subdirectory README records
coordinates, gradients, randomness, data dependencies, and checkpoint behavior.
