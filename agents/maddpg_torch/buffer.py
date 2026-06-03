"""Replay buffer for MADDPG (PyTorch)."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, max_size: int, obs_dims: Sequence[int], act_dims: Sequence[int], device: torch.device):
        self.max_size = max_size
        self.obs_dims = list(obs_dims)
        self.act_dims = list(act_dims)
        self.device = device
        self.num_agents = len(obs_dims)

        self.obs_buf = [np.zeros((max_size, d), dtype=np.float32) for d in self.obs_dims]
        self.next_obs_buf = [np.zeros((max_size, d), dtype=np.float32) for d in self.obs_dims]
        self.act_buf = [np.zeros((max_size, d), dtype=np.float32) for d in self.act_dims]
        self.rew_buf = np.zeros((max_size, self.num_agents), dtype=np.float32)
        self.done_buf = np.zeros((max_size, self.num_agents), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(
        self,
        obs: List[np.ndarray],
        acts: List[np.ndarray],
        rews: List[float],
        next_obs: List[np.ndarray],
        dones: List[bool],
    ) -> None:
        for i in range(self.num_agents):
            self.obs_buf[i][self.ptr] = obs[i]
            self.act_buf[i][self.ptr] = acts[i]
            self.next_obs_buf[i][self.ptr] = next_obs[i]
        self.rew_buf[self.ptr] = np.asarray(rews, dtype=np.float32)
        self.done_buf[self.ptr] = np.asarray(dones, dtype=np.float32)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, List[torch.Tensor], torch.Tensor]:
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = [torch.as_tensor(buf[idx], device=self.device) for buf in self.obs_buf]
        acts = [torch.as_tensor(buf[idx], device=self.device) for buf in self.act_buf]
        next_obs = [torch.as_tensor(buf[idx], device=self.device) for buf in self.next_obs_buf]
        rews = torch.as_tensor(self.rew_buf[idx], device=self.device)
        dones = torch.as_tensor(self.done_buf[idx], device=self.device)
        return obs, acts, rews, next_obs, dones

    def __len__(self) -> int:
        return self.size
