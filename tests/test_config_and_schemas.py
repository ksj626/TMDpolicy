import numpy as np
import pytest

from tmd_policy.config import HorizonConfig, load_config
from tmd_policy.data.schemas import ExpertChunk, RolloutChunk


def test_horizons_are_distinct_and_valid():
    cfg = HorizonConfig(prediction_horizon=50, execution_horizon=10)
    assert cfg.prediction_horizon == 50
    assert cfg.execution_horizon == 10
    with pytest.raises(ValueError):
        HorizonConfig(prediction_horizon=10, execution_horizon=11)


def test_config_loads_nested_dataclasses():
    cfg = load_config("configs/tiny.yaml")
    assert cfg.horizons.prediction_horizon == 50
    assert cfg.dataset.episodes == (0, 1)


def test_expert_schema_enforces_50_plan_and_10_transition_path_by_data():
    sample = ExpertChunk(
        sample_id="x",
        observation_id="o",
        dataset_id="d",
        dataset_revision="r",
        episode_index=1,
        task_index=2,
        instruction="task",
        start_frame=0,
        plan_actions=np.zeros((50, 7)),
        plan_valid=np.ones(50, dtype=bool),
        path_states=np.zeros((11, 8)),
        path_actions=np.zeros((10, 7)),
        path_valid=np.ones(10, dtype=bool),
    )
    assert sample.plan_actions.shape == (50, 7)
    assert sample.path_states.shape == (11, 8)


def test_terminal_rollout_stores_only_real_states():
    sample = RolloutChunk(
        sample_id="r",
        observation_id="o",
        policy_checkpoint="p",
        policy_version="v1",
        collection_round=1,
        task_index=0,
        instruction="task",
        chunk_index=0,
        plan_actions=np.zeros((50, 7)),
        executed_actions=np.zeros((3, 7)),
        path_states=np.zeros((4, 8)),
        path_valid=np.ones(3, dtype=bool),
        success=False,
        terminated=True,
        truncated=False,
        reset_seed=1,
        outer_noise_seed=2,
        inner_noise_seeds=(3, 4),
    )
    assert sample.path_states.shape[0] == sample.executed_actions.shape[0] + 1
