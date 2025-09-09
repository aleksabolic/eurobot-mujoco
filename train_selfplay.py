import os, argparse, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from eurobot_env import EurobotMJ
from gymnasium.wrappers import TimeLimit

def make_env(scripted_opponent=True, max_steps=1200):
    def _thunk():
        env = EurobotMJ(xml_path="assets/arena.xml", max_steps=max_steps, scripted_opponent=scripted_opponent)
        # Wrap as single-agent by exposing only blue's action; yellow controlled inside env
        from gymnasium import Env, spaces
        class BlueWrapper(Env):
            def __init__(self, e):
                self.e = e
                self.observation_space = e.observation_spaces['blue']
                self.action_space = e.action_spaces['blue']
            def reset(self, **kw):
                o, _ = self.e.reset(**kw)
                return o['blue'], _
            def step(self, a):
                obs, rew, term, trunc, info = self.e.step({'blue': a})
                return obs['blue'], rew['blue'], term['blue'], trunc['blue'], info['blue']
        return TimeLimit(BlueWrapper(env), max_episode_steps=max_steps)
    return _thunk

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--selfplay", action="store_true")
    args = ap.parse_args()

    os.makedirs("runs", exist_ok=True)
    env = make_vec_env(make_env(scripted_opponent=True), n_envs=8)
    model = PPO("MlpPolicy", env, n_steps=256, batch_size=1024, ent_coef=0.01, learning_rate=3e-4, gamma=0.995, verbose=1)
    total = 0
    ckpt_path = "runs/ppo_blue_last.zip"
    while total < args.timesteps:
        model.learn(total_timesteps=100_000, reset_num_timesteps=False)
        total += 100_000
        model.save(ckpt_path)
        print(f"Saved {ckpt_path} at {total} steps")

        if args.selfplay:
            # Reload last checkpoint as opponent: swap env to use learned opponent via scripted_opponent=False and step with model
            # For brevity, we keep scripted opponent in this minimal version.
            pass
