import os, argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import torch
from eurobot_env import EurobotDiscreteEnv
from robot import RobotProfile
from masked_policy import MaskedMultiCatPolicy
from stable_baselines3.common.callbacks import BaseCallback

torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except: pass

N = 18
CPU_IDS = list(range(N))

class FinalScoreCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._blue_scores = []
        self._yellow_scores = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        for info in infos:
            if not info:
                continue
            if "blue_final_score" in info and "yellow_final_score" in info:
                self._blue_scores.append(float(info["blue_final_score"]))
                self._yellow_scores.append(float(info["yellow_final_score"]))
        return True

    def _on_rollout_end(self) -> None:
        if self._blue_scores:
            self.logger.record(
                "rollout/blue_final_score_mean",
                float(np.mean(self._blue_scores)),
            )
            self.logger.record(
                "rollout/blue_final_score_std",
                float(np.std(self._blue_scores)),
            )
        if self._yellow_scores:
            self.logger.record(
                "rollout/yellow_final_score_mean",
                float(np.mean(self._yellow_scores)),
            )
            self.logger.record(
                "rollout/yellow_final_score_std",
                float(np.std(self._yellow_scores)),
            )
        self._blue_scores.clear()
        self._yellow_scores.clear()

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
    rl_returns = []
    score_blue = []
    score_yellow = []
    for ep in range(episodes):
        o, _ = env.reset()
        done = False
        Rb = 0.0
        steps = 0
        while not done and steps < max_steps:
            a, _ = model.predict(o, deterministic=True)
            o, r, term, trunc, info = env.step(a)
            Rb += float(r)
            done = bool(term) or bool(trunc)
            steps += 1
    
        blue_score, yellow_score = env.world.final_scores()
        rl_returns.append(Rb)
        score_blue.append(blue_score)
        score_yellow.append(yellow_score)
    mean_rl = sum(rl_returns) / max(1, len(rl_returns))
    mean_blue_score = sum(score_blue) / max(1, len(score_blue))
    mean_yellow_score = sum(score_yellow) / max(1, len(score_yellow))
    print(
        "Eval (raw): "
        f"mean_rl_return={mean_rl:.2f} mean_blue_score={mean_blue_score:.2f} "
        f"mean_yellow_score={mean_yellow_score:.2f} episodes={episodes}"
        f" rl_returns={[round(x,2) for x in rl_returns]}"
        f" blue_scores={[round(x,2) for x in score_blue]}"
        f" yellow_scores={[round(x,2) for x in score_yellow]}"
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--load-ckpt", type=str, help="Checkpoint to load before training")
    ap.add_argument("--save-ckpt", type=str, default="runs/ppo_blue_last.zip",
                    help="Checkpoint path to save after training")
    ap.add_argument("--load-vecnorm", type=str, help="VecNormalize file to load before training")
    ap.add_argument("--save-vecnorm", type=str, default="runs/vecnorm.pkl",
                    help="VecNormalize path to save after training")
    args = ap.parse_args()

    os.makedirs("runs", exist_ok=True)
    save_ckpt = args.save_ckpt
    save_vecnorm = args.save_vecnorm
    load_ckpt = args.load_ckpt or save_ckpt
    load_vecnorm = args.load_vecnorm or save_vecnorm

    ckpt_dir = os.path.dirname(save_ckpt)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)
    vecnorm_dir = os.path.dirname(save_vecnorm)
    if vecnorm_dir:
        os.makedirs(vecnorm_dir, exist_ok=True)

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

    env = DummyVecEnv(env_fns)

    # Discrete counts → keep norm_obs=False; norm_reward=True is fine.
    if load_vecnorm and os.path.exists(load_vecnorm):
        env = VecNormalize.load(load_vecnorm, env)
        env.training, env.norm_reward, env.norm_obs = True, True, False
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

    tb_log_name = os.path.splitext(os.path.basename(save_ckpt))[0] or "ppo"
    if load_ckpt and os.path.exists(load_ckpt):
        model = PPO.load(load_ckpt, env=env, device="auto")
    else:
        model = PPO(MaskedMultiCatPolicy, env,
              n_steps=2048, batch_size=36864,
              ent_coef=0.01, learning_rate=3e-4,
              gamma=0.995, clip_range=0.2, n_epochs=2,
              tensorboard_log="runs/tb",
              policy_kwargs={"mask_cfg": mask_cfg})

    total = 0
    while total < args.timesteps:
        callback = FinalScoreCallback()
        model.learn(total_timesteps=100_000, reset_num_timesteps=False,
                    progress_bar=True, tb_log_name=tb_log_name, callback=callback)
        total += 100_000
        model.save(save_ckpt)
        env.save(save_vecnorm)
        print(f"Saved {save_ckpt} at {total:,} steps")
        # Quick raw evaluation (unnormalized rewards)
        try:
            eval_policy(model, episodes=3)
        except Exception as e:
            print(f"Eval failed: {e}")
