# Student rollouts

`store.py` writes complete real LIBERO episodes. Each immutable payload contains
states `[T+1,8]`, canonical actions `[T,7]`, and per-step two-camera RGB summary
features `[T,6]`. The JSONL index stores task/instruction, reset seed, terminal
outcome, train/validation/test split, collection round, and the producing
checkpoint path plus SHA-256. Stores never contain PI0.5 KV caches.

Collection is performed by `tmd-policy rollout collect-student`; static stores
are explicitly off-policy. Current-policy occupancy experiments collect a new
round and update the recorded producer identity before retraining.

`RolloutEpisode` validates one payload, `RolloutStore.initialize/append/records`
manage its atomic tensor files and JSONL index, and `RolloutStore.validate`
checks the complete store. Environment/policy randomness is identified by reset
seed, producing checkpoint digest, and collection round; tensors are persisted
on CPU.
