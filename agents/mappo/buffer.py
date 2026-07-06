"""On-policy rollout buffer for MAPPO (PyTorch).

Unlike the off-policy replay buffers used by MAAC/MADDPG, this holds a *single*
fresh rollout of up to ``max_steps`` transitions, computes GAE advantages and
returns once the rollout is complete, then is drained over several PPO epochs and
reset. Per-agent arrays (typed by each agent's own obs/action dims) let it handle
heterogeneous environments, mirroring ``agents/maddpg_torch/buffer.py``.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        max_steps: int,
        obs_dims: Sequence[int],
        act_dims: Sequence[int],
        global_obs_dim: int,
        num_agents: int,
        device: torch.device,
    ):
        self.max_steps = max_steps
        self.obs_dims = list(obs_dims)
        self.act_dims = list(act_dims)  # storage dims: 1 (discrete index) or act_dim (continuous)
        self.global_obs_dim = global_obs_dim
        self.num_agents = num_agents
        self.device = device

        self.obs_buf = [np.zeros((max_steps, d), dtype=np.float32) for d in self.obs_dims]
        self.global_obs_buf = np.zeros((max_steps, global_obs_dim), dtype=np.float32)
        self.act_buf = [np.zeros((max_steps, d), dtype=np.float32) for d in self.act_dims]
        self.logprob_buf = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.val_buf = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.rew_buf = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.mask_buf = np.zeros((max_steps, num_agents), dtype=np.float32)  # 1 - terminated
        self.ret_buf = np.zeros((max_steps, num_agents), dtype=np.float32)
        self.adv_buf = np.zeros((max_steps, num_agents), dtype=np.float32)

        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def insert(
        self,
        obs_list: List[np.ndarray],
        act_list: List[np.ndarray],
        log_probs: List[float],
        values: List[float],
        rewards: List[float],
        masks: List[float],
    ) -> None:
        t = self.step
        for i in range(self.num_agents):
            self.obs_buf[i][t] = obs_list[i]
            self.act_buf[i][t] = act_list[i]
        self.global_obs_buf[t] = np.concatenate([np.asarray(o, dtype=np.float32).ravel() for o in obs_list])
        self.logprob_buf[t] = np.asarray(log_probs, dtype=np.float32)
        self.val_buf[t] = np.asarray(values, dtype=np.float32)
        self.rew_buf[t] = np.asarray(rewards, dtype=np.float32)
        self.mask_buf[t] = np.asarray(masks, dtype=np.float32)
        self.step += 1

    def compute_returns(self, next_value: Sequence[float], gamma: float, gae_lambda: float) -> None:
        """Generalized Advantage Estimation, computed per agent over the rollout.

        ``next_value[i]`` bootstraps agent ``i`` past the final stored step; the
        per-step ``mask`` (1 - terminated) zeroes the bootstrap on true
        terminations while still bootstrapping on time-limit truncations.
        """
        T = self.step
        for i in range(self.num_agents):
            gae = 0.0
            for t in reversed(range(T)):
                next_val = next_value[i] if t == T - 1 else self.val_buf[t + 1, i]
                mask = self.mask_buf[t, i]
                delta = self.rew_buf[t, i] + gamma * next_val * mask - self.val_buf[t, i]
                gae = delta + gamma * gae_lambda * mask * gae
                self.adv_buf[t, i] = gae
                self.ret_buf[t, i] = gae + self.val_buf[t, i]

    def agent_minibatch_generator(self, agent_i: int, num_mini_batch: int):
        """Yield shuffled minibatches for one agent's PPO update.

        Because networks are per-agent and obs/action dims are heterogeneous,
        minibatches are generated per agent over the ``T`` rollout timesteps
        (no cross-agent flattening). Advantages are normalized per agent.
        """
        T = self.step
        adv = self.adv_buf[:T, agent_i].copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        indices = np.random.permutation(T)
        mb_size = max(1, T // max(1, num_mini_batch))
        for start in range(0, T, mb_size):
            mb = indices[start:start + mb_size]
            yield {
                "obs": torch.as_tensor(self.obs_buf[agent_i][mb], device=self.device),
                "global_obs": torch.as_tensor(self.global_obs_buf[mb], device=self.device),
                "actions": torch.as_tensor(self.act_buf[agent_i][mb], device=self.device),
                "old_log_probs": torch.as_tensor(self.logprob_buf[mb, agent_i], device=self.device),
                "returns": torch.as_tensor(self.ret_buf[mb, agent_i], device=self.device),
                "advantages": torch.as_tensor(adv[mb], device=self.device),
                "values": torch.as_tensor(self.val_buf[mb, agent_i], device=self.device),
            }

    def __len__(self) -> int:
        return self.step
