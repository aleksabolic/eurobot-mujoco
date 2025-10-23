from __future__ import annotations

from typing import Iterable, Sequence, Type
from robot import Verb

import torch
import torch.nn as nn


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], activation_cls: Type[nn.Module]) -> nn.Sequential:
    layers = []
    prev_dim = int(input_dim)
    for size in hidden_sizes:
        layers.append(nn.Linear(prev_dim, int(size)))
        layers.append(activation_cls())
        prev_dim = int(size)
    return nn.Sequential(*layers)


class AlphaZeroNetwork(nn.Module):
    """Simple feed-forward backbone with separate policy and value heads."""

    def __init__(
        self,
        obs_dim: int,
        head_dims: Iterable[int] = [4,20,2,5], # dimensions of each head for each verb (same as MultiDiscrete)
        hidden_sizes: Iterable[int] = (256, 256),
        activation_cls: Type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        assert len(head_dims) == len(Verb), "teraj se u kurac"
        hidden_sizes = tuple(int(s) for s in hidden_sizes)
        self.backbone = build_mlp(obs_dim, hidden_sizes, activation_cls)
        last_dim = hidden_sizes[-1] if hidden_sizes else obs_dim

        self.policy_heads = nn.ModuleList([nn.Linear(last_dim, head_dim) for head_dim in head_dims])
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        x = self.backbone(obs)
        heads = [head(x) for head in self.policy_heads]
        logits_v, logits_n, logits_c, logits_q = heads
        value = self.value_head(x)
        return (logits_v, logits_n, logits_c, logits_q), value.squeeze(-1)

    @torch.no_grad()
    def evaluate(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(obs)
