"""Neural network modules for MADDPG (PyTorch)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActionSpec:
    is_discrete: bool
    act_dim: int
    low: np.ndarray | None = None
    high: np.ndarray | None = None


def mlp(input_dim: int, output_dim: int, hidden_dim: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_spec: ActionSpec, hidden_dim: int = 128):
        super().__init__()
        self.action_spec = action_spec
        self.net = mlp(obs_dim, action_spec.act_dim, hidden_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits_or_actions = self.net(obs)
        if self.action_spec.is_discrete:
            return logits_or_actions
        # Continuous actions in [-1, 1], will be scaled to action bounds.
        return torch.tanh(logits_or_actions)


class Critic(nn.Module):
    def __init__(self, obs_dim_total: int, act_dim_total: int, hidden_dim: int = 128):
        super().__init__()
        self.net = mlp(obs_dim_total + act_dim_total, 1, hidden_dim)

    def forward(self, obs: torch.Tensor, acts: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, acts], dim=1)
        return self.net(x)


def gumbel_softmax(logits: torch.Tensor, temperature: float = 1.0, hard: bool = False) -> torch.Tensor:
    y = F.gumbel_softmax(logits, tau=temperature, hard=hard)
    return y


def scale_continuous_action(action: torch.Tensor, low: np.ndarray, high: np.ndarray) -> torch.Tensor:
    low_t = torch.as_tensor(low, dtype=action.dtype, device=action.device)
    high_t = torch.as_tensor(high, dtype=action.dtype, device=action.device)
    return low_t + (action + 1.0) * (high_t - low_t) / 2.0
