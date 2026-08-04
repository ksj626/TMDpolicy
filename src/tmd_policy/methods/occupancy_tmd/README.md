# Occupancy-weighted TMD

`networks.py` defines a causal task-conditioned discriminator over aligned state
`[B,L,8]`, canonical action `[B,L,7]`, and two-camera RGB summaries `[B,L,6]`,
with explicit within-window positions and valid masks. Its normalizer is fitted
only from train expert and train rollout windows. The combined dataset supplies
inverse task/position/source frequency weights. Expert is label 1; student is 0.

`program.py` trains that discriminator and implements occupancy-weighted Stage-1
TMD. The density-ratio proxy is `exp(logit/temperature)`, visibly clipped to
configured bounds and detached before multiplying the per-sample MeanFlow loss.
Thus the weights alter generator gradients without training the discriminator
through the generator. Rollout producer checkpoint/round provenance determines
whether a store is current-policy or explicitly off-policy.

Public objects are `WindowNormalizer`, `OccupancyDiscriminator`,
`OccupancyDiscriminatorProgram`, `OccupancyWeightedTMDProgram`, and
`weighted_generator_loss`. The first program owns real=1/student=0 BCE; the
second reuses Stage-1 sampling and scales its per-sample gradient with detached
positive weights. Discriminator and normalizer checkpoints are immutable inputs
to occupancy-TMD.
