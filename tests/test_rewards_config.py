import pytest
from dataclasses import FrozenInstanceError

from rewards import RewardConfig


def test_reward_config_is_frozen():
    cfg = RewardConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.nest_bonus = 999


def test_reward_config_custom_values():
    cfg = RewardConfig(
        nest_bonus=5.0,
        pantry_bonus=7.0,
        interest_bonus=9.0,
        time_penalty=0.42,
        invalid_action_penalty=0.77,
    )
    assert cfg.nest_bonus == 5.0
    assert cfg.pantry_bonus == 7.0
    assert cfg.interest_bonus == 9.0
    assert cfg.time_penalty == 0.42
    assert cfg.invalid_action_penalty == 0.77
