# Canonical expert data

`libero.py` is the only expert schema. `build_episode_manifest` reads immutable
`lerobot/libero` metadata and makes task-stratified, episode-disjoint
train/validation/test lists. `LeRobotLiberoChunks` then requests action deltas
`0..49` at the dataset FPS directly from `LeRobotDataset`.

Each item contains current float images `[3,H,W]`, state `[8]`, instruction,
action `[50,7]`, `action_is_pad` and its inverse `action_valid`, episode/frame
indices, task index, and canonical task UID. Boundary actions are repeated by
LeRobot only as padding and are excluded by the boolean mask. Randomness exists
only in the seeded whole-episode split and the training sampler. Checkpoint
normalization comes from official model processors, not recomputed test data.

Public helpers are `canonical_task_uid`, `stratified_episode_split`,
`assert_episode_disjoint`, `build_episode_manifest`, and
`load_episode_manifest`. `ExpertOccupancyWindows` and
`StudentOccupancyWindows` expose real causal windows; `BalancedOccupancyDataset`
adds inverse joint task/start-position/source weights. Occupancy tensors are
float32 on CPU until the trainer moves them; masks are boolean and task/episode
identities are integer metadata.
