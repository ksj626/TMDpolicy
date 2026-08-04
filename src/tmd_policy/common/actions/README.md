# Actions

`canonical.py` defines `ActionConvention` (7-D, 50 planned, 10 executed, normalized `[-1,1]`) and validates `[B,H,D]` actions plus `[B,H]` masks. Producers are official processors; consumers are all methods and rollout. It has no side effects or trainable state.

