import numpy as np
import torch as th

from masked_policy import MaskedMultiCatPolicy
from robot import Verb
from eurobot_world import Col


def make_policy(env, mask_cfg):
    return MaskedMultiCatPolicy(
        env.observation_space,
        env.action_space,
        lr_schedule=lambda _: 0.0,
        mask_cfg=mask_cfg,
    )


def _drain_env(env, max_steps: int = 10):
    for _ in range(max_steps):
        if env.world.blue.event is None and env.world.yellow.event is None:
            return
        dt, _ = env.world._advance_until_next()
        if dt <= 0:
            break


def test_mask_shapes(env, mask_cfg):
    policy = make_policy(env, mask_cfg)
    obs, _ = env.reset(seed=0)
    obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
    masks = policy._build_masks(obs_t)
    m_verb, m_node, m_color, m_qty = masks
    assert m_verb.shape == (1, env.action_space.nvec[0])
    assert m_node.shape == (1, env.action_space.nvec[1])
    assert m_color.shape == (1, env.action_space.nvec[2])
    assert m_qty.shape == (1, env.action_space.nvec[3])


def test_pick_and_place_masks(env, mask_cfg):
    policy = make_policy(env, mask_cfg)
    env.reset(seed=1)
    pickup = env.world.PICKUPS[0]
    pantry = env.world.PANTRIES[0]

    obs, _, _, _, _ = env.step(np.array([Verb.MOVE, pickup, Col.BLUE, 0]))
    _drain_env(env)
    obs = env._obs()
    obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
    m_verb, _, _, _ = policy._build_masks(obs_t)
    assert bool(m_verb[0, Verb.PICK])

    obs, _, _, _, _ = env.step(np.array([Verb.PICK, pickup, Col.BLUE, 2]))
    _drain_env(env)
    obs = env._obs()
    assert env.world.blue.inv[Col.BLUE] >= 1

    obs, _, _, _, _ = env.step(np.array([Verb.MOVE, pantry, Col.BLUE, 0]))
    _drain_env(env)
    obs = env._obs()
    obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
    m_verb, _, _, _ = policy._build_masks(obs_t)
    assert bool(m_verb[0, Verb.PLACE])


def test_forward_and_evaluate(env, mask_cfg):
    policy = make_policy(env, mask_cfg)
    obs, _ = env.reset(seed=0)
    obs_batch = th.as_tensor(np.stack([obs, obs]), dtype=th.float32)
    actions, values, log_prob = policy.forward(obs_batch)
    assert actions.shape[0] == 2
    assert values.shape[0] == 2
    assert log_prob.shape[0] == 2

    dist = policy.get_distribution(obs_batch)
    a = dist.get_actions()
    assert a.shape == (2, env.action_space.shape[0])

    other_actions = th.as_tensor(a, dtype=th.long)
    v, logp, ent = policy.evaluate_actions(obs_batch, other_actions)
    assert v.shape[0] == 2
    assert logp.shape[0] == 2
    assert ent.shape[0] == 2
