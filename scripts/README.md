# Scripts

| File | Purpose | Public entry | Caller → consumer | Inputs → outputs | Parameters/side effects/config | Limitations |
|---|---|---|---|---|---|---|
| `generate_config_reference.py` | Regenerate or check schema-derived config documentation. | `main` | developer/CI → `tmd_policy.config.render_config_reference` | dataclass metadata → Markdown | writes `docs/config_reference.md` unless `--check`; no model config consumed | Must run with project on `PYTHONPATH`. |
| `run_diagnostics.sh` | Legacy convenience wrapper for tests/smoke. | shell script | developer → CLI/pytest | environment → diagnostic artifacts | may write its named artifact output | Prefer explicit commands in root README. |

`--check` is read-only and exits nonzero on drift. `--output` is a `Path` whose
default is inside this repository. The script contains no tensors, randomness,
trainable parameters, caches, or network calls.
