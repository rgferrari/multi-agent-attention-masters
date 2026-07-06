"""Neural network modules for MAPPO (PyTorch).

Actor is a stochastic policy (Categorical for discrete envs, diagonal Gaussian
for continuous ones). Critic is a *centralized* value function: it takes the
concatenation of every agent's observation and outputs a scalar V(s) -- the same
global-observation input used by MADDPG's critic, but value-only (no actions).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

# ``ActionSpec`` (discrete/continuous descriptor) and ``mlp`` are generic and
# already defined for MADDPG; reuse them rather than duplicating.
from agents.maddpg_torch.models import ActionSpec, mlp


class Actor(nn.Module):
    """Stochastic policy.

    Discrete: network outputs logits -> ``Categorical``.
    Continuous: network outputs the mean, paired with a state-independent
    learnable ``log_std`` -> diagonal ``Normal`` (standard PPO parameterization).
    """

    def __init__(self, obs_dim: int, action_spec: ActionSpec, hidden_dim: int = 64):
        super().__init__()
        self.action_spec = action_spec
        self.is_discrete = action_spec.is_discrete
        self.net = mlp(obs_dim, action_spec.act_dim, hidden_dim)
        if not self.is_discrete:
            self.log_std = nn.Parameter(torch.zeros(action_spec.act_dim))

    def _distribution(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        out = self.net(obs)
        if self.is_discrete:
            return torch.distributions.Categorical(logits=out)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(out, std)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample (or take the mode of) an action and return ``(action, log_prob)``."""
        dist = self._distribution(obs)
        if self.is_discrete:
            action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
            log_prob = dist.log_prob(action)
            return action, log_prob
        action = dist.mean if deterministic else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(log_prob, entropy)`` of stored ``action`` under the current policy."""
        dist = self._distribution(obs)
        if self.is_discrete:
            log_prob = dist.log_prob(action.long().squeeze(-1))
            entropy = dist.entropy()
        else:
            log_prob = dist.log_prob(action).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy


class Critic(nn.Module):
    """Centralized value function V(s) over the concatenated global observation."""

    def __init__(self, global_obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = mlp(global_obs_dim, 1, hidden_dim)

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        return self.net(global_obs)
