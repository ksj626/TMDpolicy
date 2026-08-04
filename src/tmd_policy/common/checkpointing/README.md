# Checkpointing

`method_state.py` atomically stores format-v3 model, optimizer, scheduler, scaler, counter, resolved-config, provenance, exact trainable-name, and Python/NumPy/Torch/CUDA RNG state. Strict component names prevent cross-method resume. Writes only the requested checkpoint.

