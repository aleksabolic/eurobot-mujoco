import os, argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import torch
from eurobot_env import EurobotDiscreteEnv
from robot import RobotProfile

torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except: pass

N = 8
CPU_IDS = list(range(N))

def make_env_i(i, blue: RobotProfile, yellow: RobotProfile, max_decisions=400):
    def _thunk():
        import os
        try: os.sched_setaffinity(0, {CPU_IDS[i]})
        except Exception: pass
        env = EurobotDiscreteEnv(blue_profile=blue, yellow_profile=yellow)
        return Monitor(TimeLimit(env, max_episode_steps=max_decisions))
    return _thunk

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_decisions", type=int, default=400)
    args = ap.parse_args()

    blue   = RobotProfile()          
    yellow = RobotProfile()              

    os.makedirs("runs", exist_ok=True)
    env_fns = [make_env_i(i, blue, yellow, args.max_decisions) for i in range(N)]
    env = SubprocVecEnv(env_fns, start_method="spawn")

    # Discrete counts → keep norm_obs=False; norm_reward=True is fine.
    vecnorm_path = "runs/vecnorm.pkl"
    if args.resume and os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, env)
        env.training, env.norm_reward, env.norm_obs = True, True, False
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

    ckpt_path = "runs/ppo_blue_last.zip"
    if args.resume and os.path.exists(ckpt_path):
        model = PPO.load(ckpt_path, env=env, device="auto")
    else:
        model = PPO("MlpPolicy", env,
                    n_steps=1024, batch_size=8192,
                    ent_coef=0.01, learning_rate=3e-4,
                    gamma=0.995, clip_range=0.2,
                    tensorboard_log="runs/tb")

    total = 0
    while total < args.timesteps:
        model.learn(total_timesteps=100_000, reset_num_timesteps=False,
                    progress_bar=True, tb_log_name="ppo_discrete")
        total += 100_000
        model.save(ckpt_path)
        env.save(vecnorm_path)
        print(f"Saved {ckpt_path} at {total:,} steps")
