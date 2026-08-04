# Motivation experiment

The motivating question is whether online teacher distribution matching and
transition/occupancy structure improve complete-episode success beyond official
SmolVLA Flow-SFT. Collect one evaluation JSON per baseline/retained method using
the exact same 16 suite/task entries and 20 reset seeds from
`configs/evaluation/libero_motivation.yaml` (320 paired episodes each). The main
confirmation uses all 40 suite/task entries and 50 reset seeds (2,000 each).

Use the evaluation CLI's `--policy-method`, `--checkpoint`,
`--checkpoint-sha256`, and `--output` overrides between arms; resolved outputs
still record those values. The predeclared paths in
`configs/experiments/motivation.yaml` can then be compared with
`scripts/evaluate/compare_methods.sh --output ...`. The analysis rejects
duplicate, missing, or extra pair keys and reports overall/per-suite/per-task
paired bootstrap intervals plus exact McNemar tests; individual evaluations
retain Wilson intervals, latency, and action smoothness. This directory contains
no synthetic actions, states, or claimed experimental result.

The corresponding 2,000-episode arm paths and 20,000-resample comparison are
predeclared in `configs/experiments/main.yaml`.
