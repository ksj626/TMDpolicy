# PI0.5 raw-flow parity

This real integration experiment reads one `lerobot/libero` validation batch,
runs the immutable PI0.5 model and official processor, samples fixed
`ε∈R^{1×50×32}`, and integrates `x_t` over an identical fixed descending Euler
grid. `x_t` and `v_t` are normalized PI0.5 coordinates; dimensions 0–6 are real
LIBERO actions and 7–31 are padding. `t=1` is Gaussian noise and `t=0` action.

The report compares repository and official normalized 32D samples, valid 7D
coordinates, and official postprocessed canonical actions. It also checks prefix
cache tensor identity/version, fixed-noise repeatability, velocity shape/dtype/
device, finite score statistics, and official-stat normalization round trips.
No cache is serialized. Exact command:

```bash
MUJOCO_GL=egl conda run -n tmdpolicy \
  tmd-policy teacher validate-pi05-flow \
  --config configs/teacher/pi05_flow_parity.yaml \
  --output artifacts/pi05_flow_parity
```
