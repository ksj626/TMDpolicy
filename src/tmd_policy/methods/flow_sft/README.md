# Flow-SFT

`program.py` calls the loaded checkpoint's official `SmolVLAPolicy.forward`
loss on real normalized `[B,50,7]` expert actions and terminal masks. It exposes
`head_only`, `expert_only`, `lora`, and `full`; module identities are selected
explicitly and every resulting parameter name is saved. Noise and flow time are
sampled by LeRobot on the model device. The optimizer owns only those selected
parameters, and the shared training engine checkpoints all update state.

`FlowSFTProgram` is the sole public class. `loss` delegates to the official
objective, `make_optimizers` owns only the selected names, and provenance stores
the fine-tuning mode and exact names. LeRobot owns Gaussian/time randomness;
the engine owns the global RNG state and mixed-precision context.
