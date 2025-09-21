import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from masked_policy import MaskedMultiCatPolicy
from eurobot_env import EurobotDiscreteEnv


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
