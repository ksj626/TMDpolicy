# Experiments

`motivation/` is the only current experiment package. It executes before teacher
distillation, writes raw arrays and metrics before figures, and labels synthetic
diagnostics prominently. Future real runs should add a peer package rather than
mix synthetic and robot evidence.

| File/folder | Purpose | Caller → output | Side effects/config | Limitations |
|---|---|---|---|---|
| `__init__.py` | Package marker. | Python import | none | no behavior |
| `motivation/` | M0–M5 causal discriminator and coarse-flow motivation. | experiment CLI → per-experiment run directories | reads its YAML/constants and writes under explicit output | Current saved run is synthetic only. |

Run directories use `M0/` through `M5/`, each with `metrics.json`, raw
NPZ/CSV, and both PNG and SVG where applicable. `run_summary.json` lists exactly
which experiments executed. M0 is a hard predecessor for later experiments.
