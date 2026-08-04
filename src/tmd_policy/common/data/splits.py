from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping


def split_episodes(
    episode_to_task_uid: Mapping[int, str],
    *,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[int, str]:
    if not 0 <= validation_fraction < 1 or not 0 <= test_fraction < 1:
        raise ValueError("split fractions must be in [0,1)")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation and test fractions must sum to less than one")
    grouped: dict[str, list[int]] = defaultdict(list)
    for episode, task_uid in episode_to_task_uid.items():
        grouped[task_uid].append(episode)
    rng = random.Random(seed)
    assignment: dict[int, str] = {}
    for episodes in grouped.values():
        episodes = sorted(episodes)
        rng.shuffle(episodes)
        count = len(episodes)
        n_validation = max(1, round(count * validation_fraction)) if validation_fraction and count >= 3 else 0
        n_test = max(1, round(count * test_fraction)) if test_fraction and count >= 3 else 0
        while n_validation + n_test >= count and n_test:
            n_test -= 1
        while n_validation + n_test >= count and n_validation:
            n_validation -= 1
        for episode in episodes[:n_validation]:
            assignment[episode] = "validation"
        for episode in episodes[n_validation : n_validation + n_test]:
            assignment[episode] = "test"
        for episode in episodes[n_validation + n_test :]:
            assignment[episode] = "train"
    return assignment


def assert_episode_disjoint(rows: Iterable[tuple[int, int, str]]) -> None:
    """Reject any episode (and therefore overlapping window) crossing splits.

    Rows are `(episode_index, frame_index, split)`; frame is included to make
    call sites explicit about window provenance.
    """

    by_episode: dict[int, set[str]] = defaultdict(set)
    for episode, frame, split in rows:
        if min(episode, frame) < 0:
            raise ValueError("episode/frame indices must be nonnegative")
        by_episode[episode].add(split)
    leaking = {episode: sorted(splits) for episode, splits in by_episode.items() if len(splits) > 1}
    if leaking:
        raise ValueError(f"episode-disjointness violation: {leaking}")
