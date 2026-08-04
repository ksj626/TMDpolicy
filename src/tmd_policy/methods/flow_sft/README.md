# Flow-SFT

`method.py` contains the exact pinned SmolVLA Cond-OT objective `A_t=(1-t)A+t epsilon`, target `epsilon-A`, per-sample masked reduction, freezing/LoRA/full modes, isolated optimizer/scheduler, sampling, and format-v3 resume. Inputs are officially processed `[B,50,D]`; no teacher or rollout data is consumed.

`FlowSFTConfig` defaults are `fine_tuning=frozen_backbone`,
`mixed_precision=bf16`, `gradient_accumulation=1`, `learning_rate=1e-5`, and
`weight_decay=1e-4`. Checkpoints include student, optimizer, scheduler, scaler,
RNG/cursors, config/provenance, and exact trainable names. Side effects occur
only in the explicit trainer. LoRA fails closed if no LoRA parameters exist.
