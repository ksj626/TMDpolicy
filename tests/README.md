# Tests

Tests are deterministic synthetic regressions unless a CUDA parametrization is
available; CUDA cases skip automatically when unavailable. They never step a
real environment or download a teacher.

| File | Coverage | Inputs → assertions | Side effects | Limitations |
|---|---|---|---|---|
| `test_gaussian_transition_matching.py` | Gaussian distribution/path/velocity, endpoints, residual init, masks, gradients, fixed noises, mode rejection. | synthetic tensors → algebra/shape/statistics/gradient assertions | CUDA RNG when present | Empirical Gaussian tolerance is finite-sample. |
| `test_transition_matching.py` | Outer sign/oracle, explicit ablation, evaluation counts. | synthetic chunks → endpoints/determinism | none | Toy backbone. |
| `test_config_and_schemas.py` | Strict horizons and canonical expert/rollout examples. | YAML/NumPy → dataclasses | none | Small fixtures. |
| `test_audited_systems.py` | Unknown config keys, generated docs, schema attacks, storage corruption/locking, cache collisions, resume equivalence, normalization, weights, bootstrap. | temp stores/checkpoints → hard failures/equality | writes pytest temp directory only | Does not simulate process crash mid-system-call. |
| `test_data_and_compatibility.py` | processor bridge, episode split, teacher cache/hit/inference steps. | fakes + temp store | temp files | Fake teacher; real pin checked by CLI audit. |
| `test_discriminator.py` | causal no-future leakage, variants, masks, prefix reports. | normalized synthetic paths → logit invariants | none | Does not establish real calibration. |
| `test_distillation_and_replay.py` | typed bounded mismatch and replay correction/ESS. | small vectors → semantic invariants | none | B3/B4 remains gated. |
| `test_smolvla_adapter.py` | prefix KV-cache structural integrity. | fake Smol flow/cache → equality/hard error | imports pinned LeRobot | Full official equivalence is a real smoke check. |
| `test_motivation_pipeline.py` | episode-disjoint synthetic preparation and plot reproducibility. | synthetic batches/plot data → shape/files | PNG/SVG in pytest temp | Plot aesthetics not numerically judged. |

Mathematical tensors follow the model README dictionary: action chunks
`[B,50,D]`, paths `[B,11,8]/[B,10,7]`, bool prefix masks, and per-sample loss
`[B]`. Test seeds are explicit; targets/noises never receive gradients. Run:

```bash
PYTHONPATH=src:. conda run -n lerobot pytest -q
PYTHONPATH=src:. conda run -n lerobot ruff check .
PYTHONPATH=src:. conda run -n lerobot python -m compileall -q src experiments tests
```
