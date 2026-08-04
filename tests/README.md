# Tests

Fast tests cover the installed LeRobot pin/signatures/source hashes, score
formula, cache immutability, normalization modes/round trips, differentiable
7→32 conversion, exact module selection, episode splits and terminal masks,
independent TMD Gaussian sources and the real `r=s` mixture, MeanFlow target
stop-gradient, DMD2 shared sampler/TTUR, occupancy weight propagation, and exact
checkpoint resume.

`test_real_integration.py` is separately marked and skipped unless
`TMD_RUN_INTEGRATION=1`, CUDA, an expert manifest, and cached/authenticated Hub
assets are present. It invokes the same real parity experiment as the CLI; no
synthetic test is presented as a research result.
