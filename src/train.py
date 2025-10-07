import argparse
import os
import shutil
from pathlib import Path
from typing import Optional
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit
import torch
from eurobot_env import EurobotDiscreteEnv
from eurobot_world import NEST_CAP_BLUE
from robot import RobotProfile
from masked_policy import MaskedMultiCatPolicy
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

torch.set_num_threads(1)
try: torch.set_num_interop_threads(1)
except: pass

N = 18
CPU_IDS = list(range(N))
SEED = 68

# Entropy schedule parameters
INITIAL_ENT_COEF = 0.05
FINAL_ENT_COEF = 0.01


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


class CosineEntropyCallback(BaseCallback):
    def __init__(self, initial_ent_coef: float, final_ent_coef: float, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.initial_ent_coef = float(initial_ent_coef)
        self.final_ent_coef = float(final_ent_coef)
        self.total_timesteps = max(1, int(total_timesteps))
        self._base_timestep: Optional[int] = None

    def _on_training_start(self) -> None:
        self._apply(self.model.num_timesteps)

    def _on_step(self) -> bool:
        self._apply(self.model.num_timesteps)
        return True

    def preview_value(self, current_step: int) -> float:
        return self._compute_value(current_step)

    def _apply(self, current_step: int) -> None:
        value = self._compute_value(current_step)
        _set_model_entropy_coef(self.model, value)
        self.logger.record("train/entropy_coef", float(value))

    def _compute_value(self, current_step: int) -> float:
        if self._base_timestep is None:
            self._base_timestep = current_step
        elapsed = max(current_step - self._base_timestep, 0)
        progress = min(elapsed / self.total_timesteps, 1.0)
        weight = 0.5 * (1 + np.cos(np.pi * progress))
        return float(self.final_ent_coef + (self.initial_ent_coef - self.final_ent_coef) * weight)


def _set_model_entropy_coef(model: PPO, value: float) -> None:
    model.ent_coef = float(value)
    policy = getattr(model, "policy", None)
    if policy is not None and hasattr(policy, "entropy_coef"):
        entropy_coef = getattr(policy, "entropy_coef")
        if isinstance(entropy_coef, torch.Tensor):
            entropy_coef.data.fill_(float(value))
        else:
            setattr(policy, "entropy_coef", float(value))

def make_env_i(i):
    def _thunk():
        import os
        try: os.sched_setaffinity(0, {CPU_IDS[i]})
        except Exception: pass
        env = EurobotDiscreteEnv(seed=SEED)
        return Monitor(TimeLimit(env, max_episode_steps=400))
    return _thunk

def eval_policy(model, episodes: int = 3, max_steps: int = 1000):
    """Deterministic evaluation on raw (unnormalized) env to print true returns."""
    env = EurobotDiscreteEnv(seed=SEED)
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
        f"mean_yellow_score={mean_yellow_score:.2f}"
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=100_000)
    ap.add_argument("--save-dir", type=str, default="runs/latest",
                    help="Directory where checkpoints and configs will be stored (default: runs/latest)")
    ap.add_argument("--load-dir", type=str,
                    help="Directory to load existing checkpoint and VecNormalize stats from (defaults to --save-dir)")
    args = ap.parse_args()

    save_dir = Path(args.save_dir).expanduser()
    save_dir.mkdir(parents=True, exist_ok=True)
    save_name = save_dir.name or "run"

    load_dir = Path(args.load_dir).expanduser() if args.load_dir else save_dir
    load_name = load_dir.name or save_name

    save_ckpt_path = save_dir / f"ppo_blue_{save_name}.zip"
    save_vecnorm_path = save_dir / f"vecnorm_{save_name}.pkl"
    load_ckpt_path = load_dir / f"ppo_blue_{load_name}.zip"
    load_vecnorm_path = load_dir / f"vecnorm_{load_name}.pkl"

    save_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    save_vecnorm_path.parent.mkdir(parents=True, exist_ok=True)

    robot_cfg_src = Path(__file__).resolve().parent.parent / "robot_configs"
    robot_cfg_dst = save_dir / "robot_configs"
    if robot_cfg_src.exists():
        robot_cfg_dst.mkdir(parents=True, exist_ok=True)
        for cfg_file in robot_cfg_src.iterdir():
            if cfg_file.is_file():
                shutil.copy2(cfg_file, robot_cfg_dst / cfg_file.name)

    env_fns = [make_env_i(i) for i in range(N)]

    # fetch sizes from a single env
    _tmp = EurobotDiscreteEnv(seed=SEED)
    K = len(_tmp.world.PANTRIES)
    M = len(_tmp.world.PICKUPS)
    n_nodes = _tmp.n_nodes
    max_qty = _tmp.max_qty
    tmp_world = _tmp.world
    capacity = int(tmp_world.blue_prof.capacity)
    allow_steal = bool(tmp_world.allow_steal)
    can_flip = bool(tmp_world.blue_prof.can_flip)
    pantry_cap = int(tmp_world.pantry_cap)
    nest_blue = tmp_world.NEST_BLUE
    nest_yellow = tmp_world.NEST_YELL
    # build node->local index lookups (length n_nodes)
    pantry_idx = tmp_world.pantry_idx.tolist()
    pickup_idx = tmp_world.pickup_idx.tolist()
    del _tmp
    mask_cfg = dict(
        n_pantries=K,
        n_pickups=M,
        n_nodes=n_nodes,
        max_qty=max_qty,
        capacity=capacity,
        pantry_cap=pantry_cap,
        allow_steal=allow_steal,
        can_flip=can_flip,
        pantry_idx=pantry_idx,
        pickup_idx=pickup_idx,
        pantry_nodes=tmp_world.PANTRIES,
        pickup_nodes=tmp_world.PICKUPS,
        nest_blue=nest_blue,
        nest_yellow=nest_yellow,
        nest_cap_blue=NEST_CAP_BLUE,
    )

    env = DummyVecEnv(env_fns)

    # Discrete counts → keep norm_obs=False; norm_reward=True is fine.
    if load_vecnorm_path.exists():
        env = VecNormalize.load(str(load_vecnorm_path), env)
        env.training, env.norm_reward, env.norm_obs = True, True, False
    else:
        env = VecNormalize(env, norm_obs=False, norm_reward=True, clip_obs=10.0)

    entropy_callback = CosineEntropyCallback(
        initial_ent_coef=INITIAL_ENT_COEF,
        final_ent_coef=FINAL_ENT_COEF,
        total_timesteps=args.timesteps,
    )

    tb_log_name = save_ckpt_path.stem or "ppo"
    if load_ckpt_path.exists():
        model = PPO.load(str(load_ckpt_path), env=env, device="auto")
    else:
        model = PPO(MaskedMultiCatPolicy, env,
              n_steps=2048, batch_size=36864,
              ent_coef=INITIAL_ENT_COEF, learning_rate=3e-4,
              gamma=0.995, clip_range=0.2, n_epochs=2,
              tensorboard_log="runs/tb",
              policy_kwargs={"mask_cfg": mask_cfg},
              seed=SEED)

    current_ent_coef = entropy_callback.preview_value(model.num_timesteps)
    _set_model_entropy_coef(model, current_ent_coef)

    total = 0
    while total < args.timesteps:
        callbacks = CallbackList([entropy_callback, FinalScoreCallback()])
        model.learn(total_timesteps=100_000, reset_num_timesteps=False,
                    progress_bar=True, tb_log_name=tb_log_name, callback=callbacks)
        total += 100_000
        model.save(str(save_ckpt_path))
        env.save(str(save_vecnorm_path))
        print(f"Saved {save_ckpt_path} at {total:,} steps")
        # Quick raw evaluation (unnormalized rewards)
        try:
            eval_policy(model, episodes=3)
        except Exception as e:
            print(f"Eval failed: {e}")
