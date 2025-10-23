from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import trange

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
        #TODO: add cosine lr scheduler
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        self.action_dim = action_dim
        self.obs_dim = obs_dim

    # ------------------------------------------------------------------ #
    def train(self):
        for iteration in trange(1, self.cfg.num_iterations + 1, desc="AlphaZero", unit="iter"):
            for _ in range(self.cfg.games_per_iteration):
                self._play_episode()
            self._optimize()
            self._maybe_checkpoint(iteration)

    def _optimize(self):
        if len(self.replay) < self.cfg.batch_size:
            return 
        
        self.network.train()
        for _ in range(self.cfg.training_steps):
            batch = self.replay.sample(self.cfg.batch_size)
            obs = torch.as_tensor(batch.observations, dtype=torch.float32, device=self.device)
            target_pi = torch.as_tensor(batch.policies, dtype=torch.float32, device=self.device)
            target_value = torch.as_tensor(batch.values, dtype=torch.float32, device=self.device)

            (logits_v, logits_n, logits_c, logits_q), value = self.network(obs)  # heads
            # build joint logits for ALL actions in the buffer batch
            with torch.no_grad():
                verb_idx  = torch.as_tensor(self.helper.act_verb,  device=self.device, dtype=torch.long)
                node_idx  = torch.as_tensor(self.helper.act_node,  device=self.device, dtype=torch.long)
                color_idx = torch.as_tensor(self.helper.act_color, device=self.device, dtype=torch.long)
                qty_idx   = torch.as_tensor(self.helper.act_qty,   device=self.device, dtype=torch.long)

            # logits_* are [B, dim], we want [B, A] where A=#actions
            joint_logits = (
                logits_v[:, verb_idx] +
                logits_n[:, node_idx] +
                logits_c[:, color_idx] +
                logits_q[:, qty_idx]
            )  # shape [B, A]
            log_probs = torch.log_softmax(joint_logits, dim=1)
            probs = torch.softmax(joint_logits, dim=1)

            policy_loss = -(target_pi * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value, target_value)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            loss = self.cfg.policy_loss_coef * policy_loss + self.cfg.value_loss_coef * value_loss

            self.optimizer.zero_grad()
            loss.backward()
            if self.cfg.max_grad_norm and self.cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

    def _play_episode(self):
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

        blue_score, _ = self.env.world.final_scores_norm()
        value = blue_score
        for item in episode_data:
            self.replay.add(item["obs"], item["pi"], value)

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
