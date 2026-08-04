# Real integration checks

`pi05_flow_parity.py` is the installed-package implementation of the fixed-noise
real PI0.5/LIBERO parity experiment. It downloads nothing when local-only mode is
set and assets are cached; otherwise it resolves only the immutable revisions in
configuration. The check has no training side effects.

The public `run(config_path, output_dir, sample_index=...)` function loads one
real validation item, evaluates wrapper and official samplers on identical
float32 `[1,50,32]` noise/time grids, and writes `parity.json`. CUDA/model
activations use the configured native checkpoint precision; all reported errors
are scalar host values.
