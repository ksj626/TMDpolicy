# Method contracts

These contracts are the scientific boundary of the implementation.  Code may
not use a paper method name unless its contract's capability checks pass.

| Contract | Classification | Primary sources | Default status with pinned SmolVLA / pi0.5 |
|---|---|---|---|
| `flow_sft.md` | paper-faithful action-policy reproduction | pinned LeRobot source | executable |
| `tmd.md` | paper-faithful action-space port | TM and TMD papers | Stage 1 executable; Stage 2 requires score capability |
| `dmd2_flow.md` | paper-faithful action-flow port | DMD2 paper and official code | requires teacher velocity/score adapter |
| `opd_on_policy.md` | exact categorical reproduction plus separately named continuous port | VLA-OPD paper | categorical executable; pi0.5 continuous port fails closed |
| `occupancy_tmd.md` | proposed method | repository proposal | gated on real held-out diagnostics |

The reviewed revisions were Transition Matching arXiv v1 (2506.23589), TMD
arXiv v2 (2601.09881), DMD2 arXiv v2 (2405.14867), and VLA-OPD arXiv v1
(2603.26666). DMD2's official repository is `tianweiy/DMD2`. The TMD and
VLA-OPD project pages both marked code as “coming soon” on 2026-08-04, so the
papers are authoritative for those algorithms.

Primary links: [Transition Matching](https://arxiv.org/abs/2506.23589),
[TMD](https://arxiv.org/abs/2601.09881),
[DMD2](https://arxiv.org/abs/2405.14867),
[official DMD2 code](https://github.com/tianweiy/DMD2), and
[VLA-OPD](https://arxiv.org/abs/2603.26666).
