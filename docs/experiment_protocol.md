# Experiment protocol

1. Build the immutable episode manifest and run PI0.5 fixed-noise parity.
2. Run method preflight.
3. Train DMD2 directly, or train TMD Stage 1 and let `run_tmd_pipeline.sh`
   validate/hash it before Stage 2.
4. Evaluate PI0.5 official and SmolVLA official-10 before trained arms.
5. Collect v2 student replans across all 40 tasks before occupancy training.
6. Train paper occupancy and optionally the cached-VLA adaptation.

The motivation/main comparison inputs explicitly include PI0.5 official,
SmolVLA official-10, and the four-step SmolVLA ablation so step-count effects are
visible. All arms must use identical suite/task/reset grids. Main evaluation is
all 40 tasks; use multiple training seeds for training variance and paired reset
seeds for within-checkpoint comparisons.

Static rollout data estimates historical behavior occupancy. A current-policy
claim requires a new uniquely named collection round from that exact checkpoint.
No validation/test oversampling or occupancy normalization fitting is allowed.

Minimal validation command:

```bash
conda run -n tmdpolicy pytest -q -m 'not integration'
```

Opt-in real integration commands load pinned checkpoints and one real batch only:

```bash
conda run -n tmdpolicy pytest -q -m integration
bash scripts/data/query_pi05_teacher.sh --output artifacts/pi05_flow_parity
```

Do not use these commands as performance experiments. Full training, rollout
collection, and evaluation are launched only by the explicit scripts in the root
README.
