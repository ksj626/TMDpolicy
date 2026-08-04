# Method programs

Each subpackage contains a concrete `TrainingProgram` with model graph, losses,
phase ordering, optimizer ownership, provenance, and a README. Flow-SFT consumes
expert chunks only. Stage-1 TMD consumes expert transitions and two independent
Gaussian sources. DMD2/Stage-2 additionally query the frozen online teacher and
train fake-score/GAN networks. Occupancy-discriminator consumes real expert and
student path windows; occupancy-TMD loads that immutable discriminator and
scales per-sample MeanFlow gradients. There is no method registry based on
declared strings: the CLI selects an explicit builder and fails if an asset is
missing.
