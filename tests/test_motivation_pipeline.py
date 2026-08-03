import numpy as np

from experiments.motivation.plots import plot_m2
from experiments.motivation.synthetic import generate_paths, make_splits


def test_synthetic_episode_splits_are_disjoint_task_balanced_and_canonical():
    splits = make_splits((32, 16, 24), seed=5, domain="current")
    ids = [set(batch.episode_ids.tolist()) for batch in splits.values()]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    for batch in splits.values():
        _, counts = np.unique(batch.task_ids.numpy(), return_counts=True)
        assert counts.max() - counts.min() <= 1
        assert batch.states.shape[1:] == (11, 8)
        assert batch.actions.shape[1:] == (10, 7)
        assert float(batch.actions.abs().max()) <= 1.0


def test_synthetic_controls_share_shapes_without_reusing_episodes():
    expert_a = generate_paths(12, seed=1, domain="expert", episode_offset=0)
    expert_b = generate_paths(12, seed=2, domain="expert", episode_offset=100)
    assert expert_a.states.shape == expert_b.states.shape
    assert set(expert_a.episode_ids.tolist()).isdisjoint(expert_b.episode_ids.tolist())


def test_plot_inputs_reproduce_png_and_svg_without_environment(tmp_path):
    outputs = plot_m2(
        {"success": [0, 1, 0, 1], "final_logits": [-1.0, 1.0, -0.5, 0.5]},
        tmp_path,
        "M2 | SYNTHETIC | tasks=0 | checkpoint=none | episodes=4 | split=test | "
        "seeds=1 | data=fixture.npz",
    )
    assert {tmp_path.joinpath(path).suffix for path in outputs} == {".png", ".svg"}
    assert all(tmp_path.joinpath(path).is_file() for path in outputs)
