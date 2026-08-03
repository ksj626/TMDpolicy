# Versioned data contract

Every new record has `schema_version=2`, an immutable unique `sample_id`, a safe
relative payload path, searchable JSONL provenance, and one compressed NPZ.
Actions are finite float32 canonical LIBERO coordinates in `[-1,1]`; masks are
boolean true-prefixes. Reads reject missing/corrupt payloads, duplicate IDs,
partial JSON, unsafe paths, and incompatible versions.

## Expert

| Array/metadata | Enforced contract |
|---|---|
| `plan_actions`, `plan_valid` | `(50,7)` and `(50,)`; valid prefix only. |
| `path_actions`, `path_valid` | padded `(10,7)` and `(10,)`; valid cannot exceed plan validity. |
| `path_states` | padded complete `(11,8)`, finite. |
| images | chunk-start named arrays; no future image synthesis. |
| provenance | dataset ID/revision, nonnegative episode/task/frame, language, observation ID, episode boundary flags. |

Splits use complete episodes as indivisible units and stratify by task before
overlapping windows exist. Therefore no episode/window can cross train,
validation, or test.

## Rollout

| Array/metadata | Enforced contract |
|---|---|
| `plan_actions` | full `(50,7)` postprocessed plan. |
| `executed_actions` | `(L,7)`, `0<=L<=10`, only real `env.step` actions. |
| `path_states`, `path_valid` | exactly `(L+1,8)` and all-true `(L,)`; no invented states. |
| policy provenance | checkpoint, version, collection round, task/language, chunk index. |
| stochastic provenance | reset seed, outer noise seed, all inner seeds; empty inner list only for official B0. |
| outcomes/timing | success/termination/truncation and separate preprocess/model/postprocess/environment seconds. |

CUDA is synchronized around model and postprocessing timing. Replanning occurs
at termination or immediately after ten actions.

## Teacher query

`action_chunk/action_valid` are exactly `(50,7)/(50,)` after official teacher
postprocessing. The SHA-256 key and derived immutable sample ID cover:

```text
observation_id, teacher_checkpoint, teacher_revision, processor_revision,
inference_steps, sampling_seed, sample_index
```

The configured inference-step count is applied to and verified on the actual
teacher runtime. Queries can reference only stored canonical observations and
never step an environment.

## Storage transaction

An exclusive `.writer.lock` covers duplicate detection, temporary NPZ creation,
fsync, atomic rename, and one fsynced manifest append. A read-only `audit()`
reports orphan/temp/missing/corrupt data. Explicit `recover()` can truncate only
a partial final manifest line and, with a separate opt-in, remove named orphans.
It never silently changes a schema or invents records. Old schema-less prototype
artifacts remain immutable baseline evidence and are intentionally incompatible
with schema-v2 `ChunkStore` reads.

Policy-specific normalized arrays are ephemeral. Dataset/teacher canonical
actions pass through the student's official preprocessor before a student loss;
student outputs pass through its official postprocessor before storage or the
environment. Cross-policy tensors are never compared in private normalized
coordinates.
