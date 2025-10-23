from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from alphazero.action_helper import EurobotActionHelper
from alphazero.network import AlphaZeroNetwork


@dataclass
class MCTSConfig:
    num_simulations: int = 128
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


class TreeNode:
    __slots__ = ("prior", "visit_count", "value_sum", "children", "is_expanded")

    def __init__(self, prior: float = 0.0) -> None:
        self.prior: float = float(prior)
        self.visit_count: int = 0
        self.value_sum: float = 0.0
        self.children: Dict[int, "TreeNode"] = {}
        self.is_expanded: bool = False

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTS:
    def __init__(
        self,
        network: AlphaZeroNetwork,
        action_helper: EurobotActionHelper,
        config: MCTSConfig,
        device: torch.device,
    ) -> None:
        self.network = network
        self.helper = action_helper
        self.cfg = config
        self.device = device

    def search(
        self,
        env,
        root_obs: np.ndarray,
        root_state,
        add_dirichlet_noise: bool = True,
    ) -> TreeNode:
        env.set_state(root_state)
        root = TreeNode(prior=1.0)
        self._expand(root, root_obs, env)
        if add_dirichlet_noise and root.children:
            self._add_dirichlet_noise(root)

        for _ in range(max(1, self.cfg.num_simulations)):
            env.set_state(root_state)
            self._simulate(env, root_obs, root)
        return root

    def _simulate(self, env, root_obs: np.ndarray, root: TreeNode) -> None:
        '''
        Traverse down the tree using (P)UCT and expand the first unexpanded node.
        Backprop the value back to root.
        '''
        node = root
        obs = np.array(root_obs, copy=True)
        path = [node]

        while True:
            if not node.is_expanded:
                value = self._expand(node, obs, env)
                break
            action_idx, child = self._select_child(node)
            action = self.helper.decode(action_idx)
            obs, _, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.int64)) # da li mi treba ovo??
            path.append(child)
            node = child
            done = bool(terminated) or bool(truncated)
            if done:
                value = self._terminal_value(env)
                break

        self._backprop(path, value)

    def _expand(self, node: TreeNode, obs: np.ndarray, env) -> float:
        '''
        Expands current node with all legal actions and assigns prior to all of them
        using policy network.
        '''
        mask = self.helper.legal_action_mask(obs)
        if not np.any(mask):
            print("Ne postoji validna akcija za ovo stanje!")
            node.children = {}
            node.is_expanded = True
            return self._terminal_value(env)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            (lv, ln, lc, lq), value = self.network(obs_tensor)  # each [1, dim]
        lv = lv.squeeze(0).detach().cpu()
        ln = ln.squeeze(0).detach().cpu()
        lc = lc.squeeze(0).detach().cpu()
        lq = lq.squeeze(0).detach().cpu()

        # fast composition: joint_logits[a] = lv[v[a]] + ln[n[a]] + lc[c[a]] + lq[q[a]]
        verb_idx  = torch.as_tensor(self.helper.act_verb,  dtype=torch.long)
        node_idx  = torch.as_tensor(self.helper.act_node,  dtype=torch.long)
        color_idx = torch.as_tensor(self.helper.act_color, dtype=torch.long)
        qty_idx   = torch.as_tensor(self.helper.act_qty,   dtype=torch.long)

        joint = lv[verb_idx] + ln[node_idx] + lc[color_idx] + lq[qty_idx]  # [A] torch
        joint_np = joint.numpy()

        # masked softmax over legal actions 
        priors = self._masked_softmax(joint_np, mask)
        node.children = {idx: TreeNode(prior=float(priors[idx])) for idx, valid in enumerate(mask) if valid}
        node.is_expanded = True
        return float(value.item())

    def _select_child(self, node: TreeNode) -> Tuple[int, TreeNode]:
        '''
        finds the child of the node with the best (P)UCT score
        '''
        best_score = -float("inf")
        best_action = None
        best_child = None
        sqrt_total = math.sqrt(node.visit_count + 1.0)
        for action, child in node.children.items():
            q = child.value
            u = self.cfg.c_puct * child.prior * sqrt_total / (1.0 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        if best_child is None or best_action is None:
            raise RuntimeError("MCTS: failed to select a child node. Check action masking.")
        return best_action, best_child

    def _backprop(self, path: Iterable[TreeNode], value: float) -> None:
        for node in reversed(list(path)):
            node.visit_count += 1
            node.value_sum += value

    def _add_dirichlet_noise(self, node: TreeNode) -> None:
        actions = list(node.children.keys())
        if not actions:
            return
        alpha = self.cfg.dirichlet_alpha
        epsilon = self.cfg.dirichlet_epsilon
        noise = np.random.dirichlet([alpha] * len(actions))
        for action, n in zip(actions, noise):
            child = node.children[action]
            child.prior = (1 - epsilon) * child.prior + epsilon * float(n)

    def _masked_softmax(self, logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
        masked_logits = np.where(mask, logits, -np.inf)
        finite = masked_logits[np.isfinite(masked_logits)]
        if finite.size == 0:
            probs = mask.astype(np.float32)
            total = probs.sum()
            return probs / max(total, 1.0)
        max_logit = np.max(finite)
        exp_logits = np.exp(masked_logits - max_logit)
        exp_logits = np.where(mask, exp_logits, 0.0)
        total = exp_logits.sum()
        if total <= 0.0:
            probs = mask.astype(np.float32)
            total = probs.sum()
            return probs / max(total, 1.0)
        return exp_logits / total

    def _terminal_value(self, env) -> float:
        blue_score, _ = env.world.final_scores_norm()
        return blue_score

