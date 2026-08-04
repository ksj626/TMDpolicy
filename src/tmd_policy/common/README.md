# Common infrastructure

Shared, method-neutral action, checkpoint, config, data, density, evaluation, model-freezing, processor, provenance, rollout, task, and teacher-cache contracts live here. All tensors use batch-first shapes and all stores are append-only. Side effects are limited to explicit store/checkpoint/provenance writes. See each subdirectory README and `docs/experiment_guide.md`.

