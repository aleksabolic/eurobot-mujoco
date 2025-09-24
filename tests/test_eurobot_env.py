import numpy as np
import pytest

from robot import Verb
from eurobot_world import Col


def _drain_env(env, max_steps: int = 10):
    for _ in range(max_steps):
        if env.world.blue.event is None and env.world.yellow.event is None:
            return
        dt, _ = env.world._advance_until_next()
        if dt <= 0:
            break


def test_env_spaces_consistent(env):
    action_space = env.action_space
    obs_space = env.observation_space
    assert action_space.shape == (4,)
    assert obs_space.shape[0] == env._obs_buf.shape[0]
    assert action_space.nvec[0] == 5
    assert action_space.nvec[1] == env.n_nodes
    assert action_space.nvec[2] == 3
    assert action_space.nvec[3] == env.max_qty + 1


def test_reset_observation_matches_world(env):
    obs, _ = env.reset(seed=999)
    w = env.world
    assert obs.dtype == np.float32
    assert obs[0] == pytest.approx(w.t_left * 10.0)
    assert obs[1] == pytest.approx(float(w.blue.node))
    assert obs[2] == pytest.approx(float(w.yellow.node))
    blue_inv = obs[env._sl_inv_b]
    yellow_inv = obs[env._sl_inv_y]
    assert np.allclose(blue_inv, w.blue.inv)
    assert np.allclose(yellow_inv, w.yellow.inv)


def test_step_move_then_pick(env):
    env.reset(seed=1)
    pickup = env.world.PICKUPS[0]
    # move
    obs, reward, done, _, info = env.step(np.array([Verb.MOVE, pickup, Col.BLUE, 0]))
    assert info == {}
    _drain_env(env)
    obs = env._obs()
    assert not done
    assert env.world.blue.node == pickup
    # pick
    obs, reward, done, _, info = env.step(np.array([Verb.PICK, pickup, Col.BLUE, 1]))
    assert info == {}
    _drain_env(env)
    obs = env._obs()
    assert env.world.blue.inv[Col.BLUE] >= 1
    assert env.world.pickup_avail(pickup, Col.BLUE) <= 1


def test_done_when_time_runs_out(env):
    env.reset(seed=2)
    env.world.t_left = 0.0
    obs, reward, done, _, info = env.step(np.array([Verb.WAIT, env.world.blue.node, Col.BLUE, 0]))
    assert done
    assert reward >= -10.0  # should include terminal bonus or penalty
    assert "blue_final_score" in info
    assert "yellow_final_score" in info
