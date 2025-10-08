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
    pickup = env.world.PICKUPS[0]
    env.step(np.array([Verb.PICK, pickup, Col.BLUE, 1]))
    _drain_env(env)
    obs = env._obs()
    obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
    ctx = policy._build_context(obs_t)

    verb_mask = policy._mask_verb(ctx)
    assert verb_mask.shape == (1, env.action_space.nvec[0])

    node_mask = policy._mask_node(ctx, th.tensor([Verb.PICK], dtype=th.long))
    assert node_mask.shape == (1, env.action_space.nvec[1])

    color_mask = policy._mask_color(ctx, th.tensor([Verb.PICK], dtype=th.long), ctx.blue_node.clone())
    assert color_mask.shape == (1, env.action_space.nvec[2])

    qty_mask = policy._mask_qty(ctx, th.tensor([Verb.PICK], dtype=th.long), ctx.blue_node.clone(), th.zeros(1, dtype=th.long))
    assert qty_mask.shape == (1, env.action_space.nvec[3])


def test_pick_and_place_masks(env, mask_cfg):
    policy = make_policy(env, mask_cfg)
    env.reset(seed=1)
    pickup = env.world.PICKUPS[0]
    pantry = env.world.PANTRIES[0]

    obs, _, _, _, _ = env.step(np.array([Verb.PICK, pickup, Col.BLUE, 1]))
    _drain_env(env)
    obs = env._obs()
    obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
    ctx = policy._build_context(obs_t)
    verb_mask = policy._mask_verb(ctx)
    assert bool(verb_mask[0, Verb.PICK])
    pick_node_mask = policy._mask_node(ctx, th.tensor([Verb.PICK], dtype=th.long))
    assert bool(pick_node_mask[0, pickup])

    obs, _, _, _, _ = env.step(np.array([Verb.PICK, pickup, Col.BLUE, 2]))
    _drain_env(env)
    obs = env._obs()
    assert env.world.blue.inv[Col.BLUE] >= 1

    obs = env._obs()
    obs_t = th.as_tensor(obs, dtype=th.float32).unsqueeze(0)
    ctx = policy._build_context(obs_t)
    verb_mask = policy._mask_verb(ctx)
    assert bool(verb_mask[0, Verb.PLACE])
    place_node_mask = policy._mask_node(ctx, th.tensor([Verb.PLACE], dtype=th.long))
    assert bool(place_node_mask[0, pantry])


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
