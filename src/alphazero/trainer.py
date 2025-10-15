from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from alphazero.action_helper import EurobotActionHelper
from alphazero.mcts import MCTS, MCTSConfig
from alphazero.network import AlphaZeroNetwork
from alphazero.replay_buffer import ReplayBuffer


@dataclass
class AlphaZeroConfig:
    num_iterations: int = 10
    games_per_iteration: int = 4
    num_simulations: int = 128
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    replay_buffer_size: int = 50_000
    batch_size: int = 128
    training_steps: int = 200
    value_loss_coef: float = 1.0
    policy_loss_coef: float = 1.0
    max_grad_norm: float = 5.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    temperature: float = 1.0
    temperature_decay_steps: int = 30
    device: str = "cpu"
    checkpoint_dir: Optional[Path] = None
    checkpoint_every: int = 1


class AlphaZeroTrainer:
    def __init__(self, env, network: AlphaZeroNetwork, cfg: AlphaZeroConfig) -> None:
        self.env = env
        self.cfg = cfg
        self.helper = EurobotActionHelper(env)
        obs_dim = int(env.observation_space.shape[0])
        action_dim = self.helper.size()
        if network is None:
            raise ValueError("AlphaZeroTrainer requires an initialized network.")
        self.device = torch.device(cfg.device)
        self.network = network.to(self.device)
        mcts_cfg = MCTSConfig(
            num_simulations=cfg.num_simulations,
            c_puct=cfg.c_puct,
            dirichlet_alpha=cfg.dirichlet_alpha,
            dirichlet_epsilon=cfg.dirichlet_epsilon,
        )
        self.mcts = MCTS(self.network, self.helper, mcts_cfg, self.device)
        self.replay = ReplayBuffer(cfg.replay_buffer_size)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    # ------------------------------------------------------------------ #
    def train(self) -> List[Dict[str, float]]:
        metrics_history: List[Dict[str, float]] = []
        for iteration in range(1, self.cfg.num_iterations + 1):
            print(
                f"[AlphaZero] Iteration {iteration}/{self.cfg.num_iterations}",
                flush=True,
            )
            episode_returns = []
            for game_idx in range(self.cfg.games_per_iteration):
                stats = self._play_episode()
                episode_returns.append(stats)
                print(
                    "  Self-play {}/{}: steps={} blue_score={:.1f} yellow_score={:.1f} value={:.3f}".format(
                        game_idx + 1,
                        self.cfg.games_per_iteration,
                        int(stats["steps"]),
                        stats["blue_score"],
                        stats["yellow_score"],
                        stats.get("value", 0.0),
                    ),
                    flush=True,
                )
            logs = {"iteration": float(iteration)}
            if episode_returns:
                mean_score = float(np.mean([s["blue_score"] for s in episode_returns]))
                logs["episode_blue_score"] = mean_score
                logs["episode_steps"] = float(np.mean([s["steps"] for s in episode_returns]))
                logs["episode_yellow_score"] = float(
                    np.mean([s["yellow_score"] for s in episode_returns])
                )
                print(
                    "  Rollout mean: blue={:.2f} yellow={:.2f} steps={:.1f}".format(
                        logs["episode_blue_score"],
                        logs["episode_yellow_score"],
                        logs["episode_steps"],
                    ),
                    flush=True,
                )
            train_logs = self._optimize()
            if train_logs:
                logs.update(train_logs)
                print(
                    "  Optimize: loss={:.4f} policy={:.4f} value={:.4f} entropy={:.4f} buffer={}".format(
                        train_logs.get("loss", 0.0),
                        train_logs.get("policy_loss", 0.0),
                        train_logs.get("value_loss", 0.0),
                        train_logs.get("entropy", 0.0),
                        int(train_logs.get("buffer_size", len(self.replay))),
                    ),
                    flush=True,
                )
            else:
                logs["buffer_size"] = float(len(self.replay))
                print(
                    f"  Optimize: skipped (buffer={len(self.replay)})",
                    flush=True,
                )
            metrics_history.append(logs)
            self._maybe_checkpoint(iteration)
        return metrics_history

    def _optimize(self) -> Dict[str, float]:
        logs: Dict[str, float] = {}
        if len(self.replay) < self.cfg.batch_size:
            return logs
        policy_losses = []
        value_losses = []
        total_losses = []
        entropies = []
        self.network.train()
        for _ in range(self.cfg.training_steps):
            batch = self.replay.sample(self.cfg.batch_size)
            obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
            target_pi = torch.as_tensor(batch.policies, dtype=torch.float32, device=self.device)
            target_value = torch.as_tensor(batch.values, dtype=torch.float32, device=self.device)

            logits, value = self.network(obs)
            log_probs = torch.log_softmax(logits, dim=1)
            probs = torch.softmax(logits, dim=1)
            policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value, target_value)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            loss = (
                self.cfg.policy_loss_coef * policy_loss
                + self.cfg.value_loss_coef * value_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            total_losses.append(float(loss.item()))
            entropies.append(float(entropy.item()))

        logs["policy_loss"] = float(np.mean(policy_losses))
        logs["value_loss"] = float(np.mean(value_losses))
        logs["loss"] = float(np.mean(total_losses))
        logs["entropy"] = float(np.mean(entropies))
        logs["buffer_size"] = float(len(self.replay))
        return logs

    def _play_episode(self) -> Dict[str, float]:
        self.network.eval()
        obs, _ = self.env.reset()
        done = False
        step_idx = 0
        episode_data: List[Dict[str, np.ndarray]] = []
        while not done:
            state = self.env.get_state()
            add_dirichlet = step_idx == 0 and self.cfg.dirichlet_epsilon > 0
            root = self.mcts.search(self.env, obs, state, add_dirichlet)
            visit_counts = np.zeros(self.action_dim, dtype=np.float32)
            total_visits = 0
            for action_idx, child in root.children.items():
                visit_counts[action_idx] = float(child.visit_count)
                total_visits += child.visit_count
            if total_visits <= 0:
                legal = self.helper.legal_actions(obs)
                for idx in legal:
                    visit_counts[idx] = 1.0
                total_visits = len(legal)
            policy_target = visit_counts / max(total_visits, 1)
            temperature = self._select_temperature(step_idx)
            action_idx = self._sample_action(obs, policy_target, temperature)
            action = self.helper.decode(action_idx)
            next_obs, _, term, trunc, _ = self.env.step(np.asarray(action, dtype=np.int64))
            episode_data.append(
                {"obs": np.array(obs, copy=True), "pi": policy_target, "value": 0.0}
            )
            obs = next_obs
            step_idx += 1
            done = bool(term) or bool(trunc)

        blue_score, yellow_score = self.env.world.final_scores()
        value = self._score_to_value(blue_score, yellow_score)
        for item in episode_data:
            self.replay.add(item["obs"], item["pi"], value)

        return {
            "steps": float(step_idx),
            "blue_score": float(blue_score),
            "yellow_score": float(yellow_score),
            "value": float(value),
        }


    def _score_to_value(self, blue_score: float, yellow_score: float) -> float:
        SCORE_SCALE = 160.0
        return np.tanh(blue_score / SCORE_SCALE)


    def _sample_action(self, obs: np.ndarray, policy: np.ndarray, temperature: float) -> int:
        policy = np.asarray(policy, dtype=np.float32)
        if policy.sum() <= 0 or not np.isfinite(policy).all():
            legal = np.array(self.helper.legal_actions(obs), dtype=np.int64)
            if legal.size == 0:
                raise RuntimeError("No legal actions available during action sampling.")
            return int(np.random.choice(legal))
        if temperature <= 1e-6:
            return int(np.argmax(policy))
        logits = np.log(np.clip(policy, 1e-8, None)) / max(temperature, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()
        return int(np.random.choice(len(probs), p=probs))

    def _select_temperature(self, step_idx: int) -> float:
        if self.cfg.temperature_decay_steps <= 0:
            return self.cfg.temperature
        if step_idx < self.cfg.temperature_decay_steps:
            return self.cfg.temperature
        return 1e-6

    def _maybe_checkpoint(self, iteration: int) -> None:
        directory = self.cfg.checkpoint_dir
        if directory is None or iteration % max(1, self.cfg.checkpoint_every) != 0:
            return
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"alphazero_iter{iteration:04d}.pt"
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": iteration,
            },
            path,
        )
