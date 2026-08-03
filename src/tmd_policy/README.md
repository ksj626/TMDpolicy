# Package entry points

| File | Purpose | Public API | Caller → downstream | Input → output | Trainable parameters | Side effects/config/artifacts | Limitations |
|---|---|---|---|---|---|---|---|
| `__init__.py` | Package version/export marker. | package metadata | importers | n/a | none | none | no behavior |
| `cli.py` | All audited command parsing and orchestration/gates. | `build_parser`, `main` | shell entry point → specialized modules | arguments + strict config → reports/status | command-dependent | all writes stay under explicit project output; sets project-local HF cache defaults | B3/B4 return gated status 2. |
| `config.py` | Frozen strict dataclasses, validators, resolved YAML, consumer report, generated docs. | config classes and load/save/render functions | CLI/runners/tests → constructors | YAML → `ExperimentConfig`/Markdown | none | reads YAML; writes only through `save_resolved_config` caller | Fixed audited LIBERO dimensions/horizons. |
| `smoke.py` | Tiny end-to-end Gaussian head/discriminator/replay diagnostic. | `run_synthetic_smoke` | CLI/tests → JSON report | seed/output → metrics | diagnostic head/discriminator | writes one report | Synthetic only. |

The CLI scalar namespace contains validated filesystem paths, command choices,
seeds, arm names, and optional checkpoint/store references for one process
lifetime. Configuration leaf types/ranges/provenance are exhaustive in the
generated config reference. The smoke tensors use model-standard
`[B,50,7]` actions/noises and `[B,11,8]/[B,10,7]` paths; they are CPU float32,
synthetic, masked by all-true bool prefixes, and gradients are confined to the
diagnostic models.
