"""PyTorch MADDPG implementation for MPE2 parallel envs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.maddpg_torch.buffer import ReplayBuffer
from agents.maddpg_torch.models import ActionSpec, Actor, Critic, gumbel_softmax, scale_continuous_action


@dataclass
class MADDPGConfig:
    gamma: float = 0.95
    tau: float = 0.01
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    hidden_dim: int = 128
    batch_size: int = 1024
    buffer_size: int = 1_000_000
    start_steps: int = 10_000
    update_every: int = 100
    updates_per_step: int = 1
    discrete_temperature: float = 1.0
    expl_noise: float = 0.1
    epsilon: float = 0.1


class MADDPG:
    def __init__(
        self,
        agent_ids: Sequence[str],
        obs_spaces: Sequence,
        act_spaces: Sequence,
        device: torch.device,
        config: MADDPGConfig | None = None,
    ) -> None:
        self.agent_ids = list(agent_ids)
        self.device = device
        self.config = config or MADDPGConfig()
        self.num_agents = len(self.agent_ids)

        self.obs_dims = [space.shape[0] for space in obs_spaces]
        self.action_specs = []
        for space in act_spaces:
            if hasattr(space, "n"):
                self.action_specs.append(ActionSpec(is_discrete=True, act_dim=space.n))
            else:
                low = np.asarray(space.low, dtype=np.float32)
                high = np.asarray(space.high, dtype=np.float32)
                self.action_specs.append(ActionSpec(is_discrete=False, act_dim=space.shape[0], low=low, high=high))

        self.act_dims = [spec.act_dim for spec in self.action_specs]
        self.total_obs_dim = int(sum(self.obs_dims))
        self.total_act_dim = int(sum(self.act_dims))

        self.actors: List[Actor] = []
        self.critics: List[Critic] = []
        self.target_actors: List[Actor] = []
        self.target_critics: List[Critic] = []
        self.actor_opt: List[optim.Optimizer] = []
        self.critic_opt: List[optim.Optimizer] = []

        for i in range(self.num_agents):
            actor = Actor(self.obs_dims[i], self.action_specs[i], hidden_dim=self.config.hidden_dim).to(device)
            critic = Critic(self.total_obs_dim, self.total_act_dim, hidden_dim=self.config.hidden_dim).to(device)
            target_actor = Actor(self.obs_dims[i], self.action_specs[i], hidden_dim=self.config.hidden_dim).to(device)
            target_critic = Critic(self.total_obs_dim, self.total_act_dim, hidden_dim=self.config.hidden_dim).to(device)
            target_actor.load_state_dict(actor.state_dict())
            target_critic.load_state_dict(critic.state_dict())

            self.actors.append(actor)
            self.critics.append(critic)
            self.target_actors.append(target_actor)
            self.target_critics.append(target_critic)
            self.actor_opt.append(optim.Adam(actor.parameters(), lr=self.config.actor_lr))
            self.critic_opt.append(optim.Adam(critic.parameters(), lr=self.config.critic_lr))

        self.buffer = ReplayBuffer(self.config.buffer_size, self.obs_dims, self.act_dims, device)

    def _onehot(self, action_idx: int, act_dim: int) -> np.ndarray:
        vec = np.zeros(act_dim, dtype=np.float32)
        vec[action_idx] = 1.0
        return vec

    def select_actions(self, obs: Dict[str, np.ndarray], explore: bool = True) -> Tuple[Dict[str, int | np.ndarray], List[np.ndarray]]:
        actions_env: Dict[str, int | np.ndarray] = {}
        actions_onehot: List[np.ndarray] = []
        for i, agent_id in enumerate(self.agent_ids):
            spec = self.action_specs[i]
            obs_t = torch.as_tensor(obs[agent_id], dtype=torch.float32, device=self.device).unsqueeze(0)
            logits_or_action = self.actors[i](obs_t)
            if spec.is_discrete:
                if explore and np.random.rand() < self.config.epsilon:
                    act_idx = np.random.randint(spec.act_dim)
                    act_onehot = self._onehot(act_idx, spec.act_dim)
                else:
                    act_onehot_t = gumbel_softmax(
                        logits_or_action,
                        temperature=self.config.discrete_temperature,
                        hard=True,
                    )
                    act_idx = int(act_onehot_t.argmax(dim=1).item())
                    act_onehot = act_onehot_t.squeeze(0).detach().cpu().numpy()
                actions_env[agent_id] = act_idx
                actions_onehot.append(act_onehot)
            else:
                act = logits_or_action.squeeze(0)
                if explore:
                    act = act + self.config.expl_noise * torch.randn_like(act)
                    act = torch.clamp(act, -1.0, 1.0)
                act_scaled = scale_continuous_action(act, spec.low, spec.high)
                actions_env[agent_id] = act_scaled.detach().cpu().numpy()
                actions_onehot.append(act_scaled.detach().cpu().numpy())

        return actions_env, actions_onehot

    def store_transition(
        self,
        obs: Dict[str, np.ndarray],
        actions_onehot: List[np.ndarray],
        rewards: Dict[str, float],
        next_obs: Dict[str, np.ndarray],
        dones: Dict[str, bool],
    ) -> None:
        obs_list = [obs[agent_id] for agent_id in self.agent_ids]
        next_obs_list = [next_obs[agent_id] for agent_id in self.agent_ids]
        rew_list = [rewards[agent_id] for agent_id in self.agent_ids]
        done_list = [dones[agent_id] for agent_id in self.agent_ids]
        self.buffer.add(obs_list, actions_onehot, rew_list, next_obs_list, done_list)

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for t, s in zip(target.parameters(), source.parameters()):
            t.data.copy_(t.data * (1.0 - self.config.tau) + s.data * self.config.tau)

    def update(self) -> Dict[str, float]:
        if len(self.buffer) < self.config.batch_size:
            return {}

        obs, acts, rews, next_obs, dones = self.buffer.sample(self.config.batch_size)
        obs_cat = torch.cat(obs, dim=1)
        acts_cat = torch.cat(acts, dim=1)
        next_obs_cat = torch.cat(next_obs, dim=1)

        metrics: Dict[str, float] = {}

        for i in range(self.num_agents):
            # Critic update
            with torch.no_grad():
                next_actions = []
                for j in range(self.num_agents):
                    spec = self.action_specs[j]
                    logits_or_action = self.target_actors[j](next_obs[j])
                    if spec.is_discrete:
                        next_act = gumbel_softmax(
                            logits_or_action,
                            temperature=self.config.discrete_temperature,
                            hard=True,
                        )
                    else:
                        next_act = scale_continuous_action(
                            logits_or_action,
                            spec.low,
                            spec.high,
                        )
                    next_actions.append(next_act)
                next_act_cat = torch.cat(next_actions, dim=1)
                target_q = self.target_critics[i](next_obs_cat, next_act_cat)
                y = rews[:, i : i + 1] + self.config.gamma * (1.0 - dones[:, i : i + 1]) * target_q

            q = self.critics[i](obs_cat, acts_cat)
            critic_loss = nn.functional.mse_loss(q, y)
            self.critic_opt[i].zero_grad()
            critic_loss.backward()
            self.critic_opt[i].step()

            # Actor update
            curr_actions = []
            for j in range(self.num_agents):
                spec = self.action_specs[j]
                logits_or_action = self.actors[j](obs[j])
                if spec.is_discrete:
                    curr_act = gumbel_softmax(
                        logits_or_action,
                        temperature=self.config.discrete_temperature,
                        hard=True,
                    )
                else:
                    curr_act = scale_continuous_action(
                        logits_or_action,
                        spec.low,
                        spec.high,
                    )
                if j != i:
                    curr_act = curr_act.detach()
                curr_actions.append(curr_act)
            curr_act_cat = torch.cat(curr_actions, dim=1)
            actor_loss = -self.critics[i](obs_cat, curr_act_cat).mean()
            self.actor_opt[i].zero_grad()
            actor_loss.backward()
            self.actor_opt[i].step()

            # Soft update targets
            self._soft_update(self.target_critics[i], self.critics[i])
            self._soft_update(self.target_actors[i], self.actors[i])

            metrics[f"critic_loss_{i}"] = float(critic_loss.item())
            metrics[f"actor_loss_{i}"] = float(actor_loss.item())

        return metrics

    def state_dict(self) -> Dict[str, object]:
        return {
            "actors": [a.state_dict() for a in self.actors],
            "critics": [c.state_dict() for c in self.critics],
            "target_actors": [a.state_dict() for a in self.target_actors],
            "target_critics": [c.state_dict() for c in self.target_critics],
            "actor_opt": [o.state_dict() for o in self.actor_opt],
            "critic_opt": [o.state_dict() for o in self.critic_opt],
            "config": self.config.__dict__,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        for a, sd in zip(self.actors, state.get("actors", [])):
            a.load_state_dict(sd)
        for c, sd in zip(self.critics, state.get("critics", [])):
            c.load_state_dict(sd)
        for a, sd in zip(self.target_actors, state.get("target_actors", [])):
            a.load_state_dict(sd)
        for c, sd in zip(self.target_critics, state.get("target_critics", [])):
            c.load_state_dict(sd)
        for o, sd in zip(self.actor_opt, state.get("actor_opt", [])):
            o.load_state_dict(sd)
        for o, sd in zip(self.critic_opt, state.get("critic_opt", [])):
            o.load_state_dict(sd)
