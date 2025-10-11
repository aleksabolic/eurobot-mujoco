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
    assert action_space.nvec[0] == len(Verb)
    assert action_space.nvec[1] == env.n_nodes
    assert action_space.nvec[2] == len(Col)
    assert action_space.nvec[3] == env.max_qty + 1


def test_reset_observation_matches_world(env):
    obs, _ = env.reset(seed=999)
    w = env.world
    assert obs.dtype == np.float32
    blue_one_hot = obs[env._sl_node_blue]
    yellow_one_hot = obs[env._sl_node_yellow]
    assert blue_one_hot.sum() == pytest.approx(1.0)
    assert yellow_one_hot.sum() == pytest.approx(1.0)
    assert int(np.argmax(blue_one_hot)) == int(w.blue.node)
    assert int(np.argmax(yellow_one_hot)) == int(w.yellow.node)
    blue_inv = obs[env._sl_inv_b]
    assert np.allclose(blue_inv, w.blue.inv)


def test_step_reposition_then_pick(env):
    env.reset(seed=1)
    pickup = env.world.PICKUPS[0]
    # reposition via zero-qty pick
    obs, reward, done, _, info = env.step(np.array([Verb.PICK, pickup, Col.BLUE, 1]))
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
    obs, reward, done, _, info = env.step(np.array([Verb.PLACE, env.world.blue.node, Col.BLUE, 1]))
    assert done
    assert reward >= -10.0  # should include terminal bonus or penalty
    assert "blue_final_score" in info
    assert "yellow_final_score" in info
