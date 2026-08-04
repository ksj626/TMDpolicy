# Density

`schedule.py` implements Cond-OT interpolation and the conditional-marginal velocity-to-score identity on explicit nonsingular time support. `cnf.py` integrates continuous-flow log density with separately named exact full trace and replayable Hutchinson estimator. Inputs are `[B,...]`; exact mode is correctness-first and expensive in dimension.

