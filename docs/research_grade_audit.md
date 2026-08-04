# Research-grade refactor audit

## Confirmed pre-refactor defects

| Severity | Finding | Repair requirement |
|---|---|---|
| P0 | Expert episode 0, early rollout task 0, and LIBERO-10 environment task 0 refer to different instructions. | Canonical `TaskIdentity`; reject joins on integer indices alone. |
| P0 | Plain Gaussian residual flow was labeled generically as TMD and exposed source noise to the head. | Rename as ablation; implement TM-MF separately; remove source noise from head inputs. |
| P0 | π0.5 exposes no supported normalized log-probability API. | OPD fails closed; no action-MSE surrogate. |
| P1 | LeRobot has the expected commit but a dirty worktree. | Save exact dependency patch/hash in provenance or refuse the run. |
| P1 | Existing schemas use one `task_index` namespace. | Store suite, dataset, episode, BDDL, version, and canonical UID fields. |
| P1 | Existing expert trainer selects one manifest record. | Multi-record dataset/DataLoader and episode-disjoint split enforcement. |
| P1 | Payload manifest lacks a content digest. | Hash canonical metadata and payload bytes. |
| P1 | Image arrays lack complete dtype/layout/range/schema checks. | Fail-closed image validator. |
| P2 | Termination state and local time limit are partially conflated. | Store task success, terminated, environment truncated, and local limit separately. |
| P2 | Metric naming did not distinguish trapezoidal PR-AUC and average precision. | Separate implementations and names. |

Historical artifacts remain immutable evidence but are schema-v2 and are not
joinable with new schema-v3 data until a task-identity migration explicitly
resolves each record.
