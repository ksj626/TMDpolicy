# Configurations

`tiny.yaml` is the smallest audited smoke/B2 configuration.
`first_experiment.yaml` expands expert episodes and update counts while retaining
the exact model/data revisions, LIBERO dimensions, mode, tasks, and seeds.
Unknown keys fail recursively; unsupported values fail dataclass validation.
Every command persists the fully resolved defaults and values in its run folder.

| File | Purpose | Consumer | Output/side effects | Limitations |
|---|---|---|---|---|
| `tiny.yaml` | Unit/smoke and one-chunk B2 settings. | all CLI commands via `load_config` | copied as resolved YAML to run output | Not a powered robot experiment. |
| `first_experiment.yaml` | B0/B1/B2 task/seed comparison settings. | training/evaluation runners | copied to every arm | B3/B4 fields remain gated. |

All leaf fields, types, defaults, mathematical meanings, ranges, consumers,
cache/checkpoint invalidation, affected phases, and recommendations are generated
from dataclass metadata in [the config reference](../docs/config_reference.md).
Run `PYTHONPATH=src python scripts/generate_config_reference.py --check` in CI.

Important scalar groups are horizons `(50,10)`, canonical dimensions `(8,7)`,
pinned checkpoint/dataset/processor/LeRobot revisions, Gaussian TM architecture
and solver counts, discriminator width/depth/weight bounds, and training
tasks/seeds/optimizer/replay policy. Lists must be nonempty, unique, and
nonnegative; dropout/probabilities and optimizer ranges are validated.
