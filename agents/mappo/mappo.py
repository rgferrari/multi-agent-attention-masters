"""PyTorch MAPPO (Multi-Agent PPO) implementation for parallel envs.

On-policy counterpart to this repo's MAAC/MADDPG agents. Each agent has its own
actor and its own *centralized* critic (fed the concatenated global observation),
following the per-agent-independent-networks convention of ``agents/maddpg_torch``
rather than upstream MAPPO's parameter sharing. A single class handles both
discrete (MPE2) and continuous (MAMuJoCo) action spaces, branching on the
per-agent ``ActionSpec`` -- so no separate continuous variant is needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.maddpg_torch.models import ActionSpec
from agents.mappo.models import Actor, Critic


@dataclass
class MAPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    ppo_epoch: int = 15
    num_mini_batch: int = 1
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    max_grad_norm: float = 10.0
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    hidden_dim: int = 64
    use_clipped_value_loss: bool = True


class MAPPO:
    def __init__(
        self,
        agent_ids: Sequence[str],
        obs_spaces: Sequence,
        act_spaces: Sequence,
        device: torch.device,
        config: MAPPOConfig | None = None,
    ) -> None:
        self.agent_ids = list(agent_ids)
        self.device = device
        self.config = config or MAPPOConfig()
        self.num_agents = len(self.agent_ids)

        self.obs_dims = [space.shape[0] for space in obs_spaces]
        self.action_specs: List[ActionSpec] = []
        for space in act_spaces:
            if hasattr(space, "n"):
                self.action_specs.append(ActionSpec(is_discrete=True, act_dim=space.n))
            else:
                low = np.asarray(space.low, dtype=np.float32)
                high = np.asarray(space.high, dtype=np.float32)
                self.action_specs.append(ActionSpec(is_discrete=False, act_dim=space.shape[0], low=low, high=high))

        # Storage dim per agent action: a single index for discrete, the full
        # vector for continuous. Exposed so the training loop can size the buffer.
        self.act_store_dims = [1 if spec.is_discrete else spec.act_dim for spec in self.action_specs]
        self.total_obs_dim = int(sum(self.obs_dims))

        self.actors: List[Actor] = []
        self.critics: List[Critic] = []
        self.actor_opt: List[optim.Optimizer] = []
        self.critic_opt: List[optim.Optimizer] = []
        for i in range(self.num_agents):
            actor = Actor(self.obs_dims[i], self.action_specs[i], hidden_dim=self.config.hidden_dim).to(device)
            critic = Critic(self.total_obs_dim, hidden_dim=self.config.hidden_dim).to(device)
            self.actors.append(actor)
            self.critics.append(critic)
            self.actor_opt.append(optim.Adam(actor.parameters(), lr=self.config.actor_lr))
            self.critic_opt.append(optim.Adam(critic.parameters(), lr=self.config.critic_lr))

    def _global_obs(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        flat = np.concatenate([np.asarray(obs[a], dtype=np.float32).ravel() for a in self.agent_ids])
        return torch.as_tensor(flat, dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_actions(
        self, obs: Dict[str, np.ndarray], deterministic: bool = False
    ) -> Tuple[Dict[str, int | np.ndarray], List[np.ndarray], List[float], List[float]]:
        """Roll out one step.

        Returns ``(actions_env, actions_store, log_probs, values)``:
        ``actions_env`` is env-ready (int for discrete, clipped vector for
        continuous); ``actions_store`` is what the buffer keeps (the raw sampled
        action, so PPO ratios stay consistent).
        """
        actions_env: Dict[str, int | np.ndarray] = {}
        actions_store: List[np.ndarray] = []
        log_probs: List[float] = []
        values: List[float] = []
        global_obs = self._global_obs(obs)

        for i, agent_id in enumerate(self.agent_ids):
            spec = self.action_specs[i]
            obs_t = torch.as_tensor(obs[agent_id], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob = self.actors[i].get_action(obs_t, deterministic=deterministic)
                value = self.critics[i](global_obs)

            if spec.is_discrete:
                act_idx = int(action.item())
                actions_env[agent_id] = act_idx
                actions_store.append(np.array([act_idx], dtype=np.float32))
            else:
                act_vec = action.squeeze(0).cpu().numpy().astype(np.float32)
                actions_env[agent_id] = np.clip(act_vec, spec.low, spec.high)
                actions_store.append(act_vec)
            log_probs.append(float(log_prob.item()))
            values.append(float(value.item()))

        return actions_env, actions_store, log_probs, values

    def get_values(self, obs: Dict[str, np.ndarray]) -> List[float]:
        """Bootstrap value V(s) per agent from the (final) observation."""
        global_obs = self._global_obs(obs)
        values: List[float] = []
        with torch.no_grad():
            for i in range(self.num_agents):
                values.append(float(self.critics[i](global_obs).item()))
        return values

    def update(self, buffer) -> Dict[str, float]:
        """PPO update: per agent, several epochs of clipped-surrogate minibatch SGD."""
        cfg = self.config
        metrics: Dict[str, float] = {}

        for i in range(self.num_agents):
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
                    actor_loss = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * entropy.mean()

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

            n = max(1, len(actor_losses))
            metrics[f"actor_loss_{i}"] = sum(actor_losses) / n
            metrics[f"value_loss_{i}"] = sum(value_losses) / n
            metrics[f"entropy_{i}"] = sum(entropies) / n

        return metrics

    def state_dict(self) -> Dict[str, object]:
        return {
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics],
            "actor_opt": [o.state_dict() for o in self.actor_opt],
            "critic_opt": [o.state_dict() for o in self.critic_opt],
            "config": self.config.__dict__,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        for a, sd in zip(self.actors, state.get("actors", [])):
            a.load_state_dict(sd)
        for c, sd in zip(self.critics, state.get("critics", [])):
            c.load_state_dict(sd)
        for o, sd in zip(self.actor_opt, state.get("actor_opt", [])):
            o.load_state_dict(sd)
        for o, sd in zip(self.critic_opt, state.get("critic_opt", [])):
            o.load_state_dict(sd)
