import os, argparse, time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from eurobot_env import EurobotMJ
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm

import torch
torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except: pass

N = 8 
CPU_IDS = list(range(8)) 

def make_env_i(i, scripted_opponent=True, max_steps=2000):
    def _thunk():
        import os
        try:
            os.sched_setaffinity(0, {CPU_IDS[i]}) 
        except Exception:
            pass
        env = EurobotMJ(xml_path="assets/arena.xml", max_steps=max_steps, scripted_opponent=scripted_opponent)
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
        from gymnasium.wrappers import TimeLimit
        wrapped = TimeLimit(BlueWrapper(env), max_episode_steps=max_steps)
        return Monitor(wrapped)
    return _thunk


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--selfplay", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs("runs", exist_ok=True)
    env = SubprocVecEnv([make_env_i(i) for i in range(N)], start_method="spawn")

    if args.resume and os.path.exists("runs/vecnorm.pkl"):
        env = VecNormalize.load("runs/vecnorm.pkl", env)
        env.training = True
        env.norm_reward = True
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    ckpt_path = "runs/ppo_blue_last.zip"

    if args.resume and os.path.exists(ckpt_path):
        model = PPO.load(ckpt_path, env=env, device="auto")
    else:
        model = PPO(
            "MlpPolicy", env,
            n_steps=1024, batch_size=8192,
            ent_coef=0.01, learning_rate=3e-4,
            gamma=0.99, clip_range=0.2,
            tensorboard_log="runs/tb", verbose=0
        )

    total = 0
    while total < args.timesteps:
        model.learn(
            total_timesteps=100_000,
            reset_num_timesteps=False, 
            progress_bar=True,
            tb_log_name="ppo_resume"
        )
        total += 100_000
        model.save(ckpt_path)
        env.save("runs/vecnorm.pkl")
        print(f"Saved {ckpt_path} at {total} steps")

        if args.selfplay:
            # Reload last checkpoint as opponent: swap env to use learned opponent via scripted_opponent=False and step with model
            # For brevity, we keep scripted opponent in this minimal version.
            pass
