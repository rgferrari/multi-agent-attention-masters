"""PyTorch HAPPO (Heterogeneous-Agent PPO) implementation for parallel envs.

HAPPO (Kuba et al., 2022, "Trust Region Policy Optimisation in Multi-Agent
Reinforcement Learning") fixes a problem MAPPO has: when every agent updates
simultaneously, each one optimises as if its teammates stood still. That
assumption is false, so the joint improvement can be worse than any individual
agent's -- there is no monotonic improvement guarantee.

HAPPO applies the *multi-agent advantage decomposition theorem* to update agents
**sequentially, in a random order**. Each agent's surrogate objective is scaled
by a ``factor`` accumulating how much the already-updated agents changed their
policies:

    factor <- factor * exp(new_log_prob - old_log_prob)

so agent ``i`` optimises against the teammates' *post-update* behaviour rather
than their stale behaviour. This restores the monotonic improvement guarantee
without requiring parameter sharing -- hence "heterogeneous-agent": policies may
differ in architecture and in observation/action dimensions.

Relation to the other algorithms here: HAPPO and MAT rest on the *same theorem*
but apply it in opposite places. HAPPO is sequential in the **optimisation
procedure** (update one policy at a time); MAT is sequential in the
**architecture** (its decoder emits actions autoregressively over agents). MAPPO
applies the theorem not at all. That makes HAPPO the natural bridge between the
two in this repo's comparison.

Networks are inherited unchanged from ``agents/mappo`` -- per-agent actor plus
per-agent centralized critic over the concatenated global observation. Only the
update rule differs, so this subclasses ``MAPPO`` rather than duplicating the
rollout, checkpoint, and construction logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from agents.mappo.mappo import MAPPO, MAPPOConfig


@dataclass
class HAPPOConfig(MAPPOConfig):
    """Same knobs as MAPPO -- HAPPO changes the update procedure, not the
    hyperparameter surface."""


class HAPPO(MAPPO):
    def __init__(self, *args, config: HAPPOConfig | None = None, **kwargs) -> None:
        super().__init__(*args, config=config or HAPPOConfig(), **kwargs)

    def _log_probs_for(self, agent_i: int, buffer) -> torch.Tensor:
        """Log-probs of the stored actions under agent ``agent_i``'s current policy.

        Evaluated over the whole rollout in one pass (no minibatching) so the
        before/after comparison that forms the factor is exact.
        """
        obs = torch.as_tensor(buffer.obs_buf[agent_i][: buffer.step], device=self.device)
        actions = torch.as_tensor(buffer.act_buf[agent_i][: buffer.step], device=self.device)
        with torch.no_grad():
            log_probs, _ = self.actors[agent_i].evaluate(obs, actions)
        return log_probs

    def update(self, buffer) -> Dict[str, float]:
        """Sequential HAPPO update over a random agent permutation.

        Differs from ``MAPPO.update`` in exactly two ways:
        * agents are updated one at a time in a random order, and
        * each agent's surrogate is scaled by ``factor``, the cumulative policy
          change of the agents already updated this iteration.
        """
        cfg = self.config
        metrics: Dict[str, float] = {}

        # factor[t] multiplies the surrogate at rollout step t. Starts at 1: the
        # first agent in the order faces the same objective MAPPO would give it.
        factor = torch.ones(buffer.step, dtype=torch.float32, device=self.device)

        # Random order each iteration, as in the reference implementation --
        # a fixed order would systematically privilege whoever updates first.
        for i in torch.randperm(self.num_agents).tolist():
            old_log_probs = self._log_probs_for(i, buffer)

            actor_losses: List[float] = []
            value_losses: List[float] = []
            entropies: List[float] = []

            for _ in range(cfg.ppo_epoch):
                for mb in buffer.agent_minibatch_generator(i, cfg.num_mini_batch):
                    log_probs, entropy = self.actors[i].evaluate(mb["obs"], mb["actions"])
                    values = self.critics[i](mb["global_obs"]).squeeze(-1)

                    ratio = torch.exp(log_probs - mb["old_log_probs"])
                    surr1 = ratio * mb["advantages"]
                    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * mb["advantages"]
                    # The HAPPO term: scale by how much earlier agents moved.
                    actor_loss = (
                        -(mb["factor"] * torch.min(surr1, surr2)).mean()
                        - cfg.entropy_coef * entropy.mean()
                    )

                    returns = mb["returns"]
                    if cfg.use_clipped_value_loss:
                        old_values = mb["values"]
                        v_clipped = old_values + torch.clamp(values - old_values, -cfg.clip_param, cfg.clip_param)
                        value_loss = 0.5 * torch.max((values - returns) ** 2, (v_clipped - returns) ** 2).mean()
                    else:
                        value_loss = 0.5 * ((values - returns) ** 2).mean()

                    self.actor_opt[i].zero_grad()
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), cfg.max_grad_norm)
                    self.actor_opt[i].step()

                    self.critic_opt[i].zero_grad()
                    (value_loss * cfg.value_loss_coef).backward()
                    nn.utils.clip_grad_norm_(self.critics[i].parameters(), cfg.max_grad_norm)
                    self.critic_opt[i].step()

                    actor_losses.append(float(actor_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy.mean().item()))

            # Accumulate this agent's policy change into the factor the *next*
            # agent in the order will see.
            new_log_probs = self._log_probs_for(i, buffer)
            factor = factor * torch.exp(new_log_probs - old_log_probs)
            buffer.set_factor(factor.detach().cpu().numpy())

            n = max(1, len(actor_losses))
            metrics[f"actor_loss_{i}"] = sum(actor_losses) / n
            metrics[f"value_loss_{i}"] = sum(value_losses) / n
            metrics[f"entropy_{i}"] = sum(entropies) / n

        metrics["factor_mean"] = float(factor.mean().item())
        return metrics
