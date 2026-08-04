# Implementation report

The former scaffold was replaced with installed-package LeRobot compatibility,
repository-owned PI0.5/SmolVLA backends, one real LIBERO schema, executable
resumable trainers for all retained methods, real rollout/occupancy paths, a
fixed-noise PI0.5 parity experiment, and expanded paired LIBERO protocols.

Removed paths include the complete VLA-OPD method directory, two OPD configs,
its training script/documentation, capability registry/research CLI, OPD-only
CNF/density code, duplicate schema-v2/v3 stores, synthetic motivation runner,
and stale gated coordinator code. PyPI LeRobot 0.6.1 is now external and source
hashed; no Git checkout or `.git` directory is required.

Affordable unit/import/static validation is run in Conda `tmdpolicy`. Full model
training, downloads, complete rollouts, main evaluation, and multi-GPU work are
intentionally not launched. The real parity test is separately gated on CUDA,
authentication, checkpoint, dataset videos, and the built episode manifest.

The final local validation on 2026-08-04 passed the CUDA/BF16/Linux/Python and
pinned LeRobot import/signature/source-hash verifier, 23 fast tests, Python
compilation, shell syntax, YAML loading, CLI help, and `git diff --check`. One
real PI0.5/LIBERO integration test remained gated because the immutable dataset
manifest/videos and complete model assets were not locally prepared; no large
download was initiated.

Known limits are truthful: the primary TMD transformer is a paper-closest
SmolVLA action-space port rather than an exact paper-native split; the default
DMD fake-score network is a practical adaptation; PI0.5 clone memory is large;
occupancy-TMD is proposed; dependent checkpoint SHA placeholders can only be
filled after upstream runs; full performance and memory estimates remain to be
measured on the target hardware.
