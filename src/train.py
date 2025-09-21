import os, argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import torch
from eurobot_env import EurobotDiscreteEnv
from robot import RobotProfile
from masked_policy import MaskedMultiCatPolicy

torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except: pass

N = 18
CPU_IDS = list(range(N))

def make_env_i(i):
    def _thunk():
        import os
        try: os.sched_setaffinity(0, {CPU_IDS[i]})
        except Exception: pass
        env = EurobotDiscreteEnv()
        return Monitor(TimeLimit(env, max_episode_steps=400))
    return _thunk

def eval_policy(model, episodes: int = 3, max_steps: int = 1000):
    """Deterministic evaluation on raw (unnormalized) env to print true returns."""
    env = EurobotDiscreteEnv()
    returns = []
    for ep in range(episodes):
        o, _ = env.reset()
        done = False
        R = 0.0
        steps = 0
        while not done and steps < max_steps:
            a, _ = model.predict(o, deterministic=True)
            o, r, term, trunc, _ = env.step(a)
            R += float(r)
            done = bool(term) or bool(trunc)
            steps += 1
        returns.append(R)
    mean_R = sum(returns) / max(1, len(returns))
    print(f"Eval (raw): mean={mean_R:.2f} episodes={episodes} returns={[round(x,2) for x in returns]}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()       

    os.makedirs("runs", exist_ok=True)
    env_fns = [make_env_i(i) for i in range(N)]

    # fetch sizes from a single env
    _tmp = EurobotDiscreteEnv()
    K = len(_tmp.world.PANTRIES)
    M = len(_tmp.world.PICKUPS)
    n_nodes = _tmp.n_nodes
    max_qty = _tmp.max_qty
    capacity = int(_tmp.world.blue_prof.capacity)
    allow_steal = bool(_tmp.world.allow_steal)
    can_flip = bool(_tmp.world.blue_prof.can_flip)
    # build node->local index lookups (length n_nodes)
    pantry_idx = _tmp.world.pantry_idx.tolist()
    pickup_idx = _tmp.world.pickup_idx.tolist()
    del _tmp
    mask_cfg = dict(
        n_pantries=K, n_pickups=M, n_nodes=n_nodes, max_qty=max_qty,
        capacity=capacity, allow_steal=allow_steal, can_flip=can_flip,
        pantry_idx=pantry_idx, pickup_idx=pickup_idx,
    )

    env = SubprocVecEnv(env_fns)

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
        model = PPO(MaskedMultiCatPolicy, env,
              n_steps=4096, batch_size=36864,
              ent_coef=0.01, learning_rate=3e-4,
              gamma=0.995, clip_range=0.2, n_epochs=5,
              tensorboard_log="runs/tb",
              policy_kwargs={"mask_cfg": mask_cfg})

    total = 0
    while total < args.timesteps:
        model.learn(total_timesteps=100_000, reset_num_timesteps=False,
                    progress_bar=True, tb_log_name="ppo_discrete")
        total += 100_000
        model.save(ckpt_path)
        env.save(vecnorm_path)
        print(f"Saved {ckpt_path} at {total:,} steps")
        # Quick raw evaluation (unnormalized rewards)
        try:
            eval_policy(model, episodes=3)
        except Exception as e:
            print(f"Eval failed: {e}")
