import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from masked_policy import MaskedMultiCatPolicy
from eurobot_env import EurobotDiscreteEnv
from eurobot_world import Col
from robot import Verb


def _force_policy_action(policy, action):
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.fill_(-100.0)
        offset = 0
        for head_idx, choice in enumerate(action):
            head_size = int(policy.nvec[head_idx].item())
            bias_slice = policy.action_net.bias[offset : offset + head_size]
            bias_slice.fill_(-100.0)
            bias_slice[int(choice)] = 100.0
            offset += head_size


def _drain_world(world, max_steps: int = 16):
    for _ in range(max_steps):
        if world.blue.event is None and world.yellow.event is None:
            return
        dt, _ = world._advance_until_next()
        if dt <= 0:
            break


def test_short_ppo_rollout(mask_cfg):
    def _make_env():
        def _init():
            env = EurobotDiscreteEnv(seed=0)
            return env
        return _init

    vec_env = DummyVecEnv([_make_env()])
    model = PPO(
        MaskedMultiCatPolicy,
        vec_env,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        gamma=0.95,
        learning_rate=3e-4,
        policy_kwargs={"mask_cfg": mask_cfg},
    )
    model.learn(total_timesteps=64)

    obs = vec_env.reset()
    action, _ = model.predict(obs, deterministic=False)
    assert action.shape[-1] == 4


def test_ppo_pick_flip_place_sequence(mask_cfg):
    def _make_env():
        def _init():
            return EurobotDiscreteEnv(seed=7)
        return _init

    vec_env = DummyVecEnv([_make_env()])
    env = vec_env.envs[0]

    class _IdlePolicy:
        def next_action(self, actor_tag, world, state, rng):
            return (int(Verb.WAIT), int(state.node), int(Col.YELLOW), 1)

    env.world.yellow_policy = _IdlePolicy()

    model = PPO(
        MaskedMultiCatPolicy,
        vec_env,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        gamma=0.95,
        learning_rate=3e-4,
        policy_kwargs={"mask_cfg": mask_cfg},
    )

    obs = vec_env.reset()

    pickup_node = env.world.PICKUPS[0]
    pantry_node = env.world.PANTRIES[0]
    pickup_local = int(env.world.pickup_idx[pickup_node])
    pantry_local = int(env.world.pantry_idx[pantry_node])

    action_plan = [
        np.array([int(Verb.MOVE), pickup_node, int(Col.YELLOW), 0]),
        np.array([int(Verb.PICK), pickup_node, int(Col.YELLOW), 1]),
        np.array([int(Verb.FLIP), pickup_node, int(Col.BLUE), 1]),
        np.array([int(Verb.MOVE), pantry_node, int(Col.BLUE), 0]),
        np.array([int(Verb.PLACE), pantry_node, int(Col.BLUE), 1]),
    ]

    for expected in action_plan:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        m_verb, _, _, _ = model.policy._build_masks(obs_tensor)
        assert bool(m_verb[0, expected[0]])
        _force_policy_action(model.policy, expected)
        action, _ = model.predict(obs, deterministic=True)
        assert action.shape == (1, 4)
        np.testing.assert_array_equal(action[0], expected)
        obs, rewards, dones, infos = vec_env.step(action)
        assert not bool(dones[0])
        _drain_world(env.world)
        obs = env._obs().astype(np.float32)[None, :]

    assert int(env.world.pickups[pickup_local, Col.YELLOW]) == 1
    assert int(env.world.pantries[pantry_local, Col.BLUE]) == 1
    assert int(env.world.blue.inv.sum()) == 0
    assert int(env.world.blue.node) == pantry_node
