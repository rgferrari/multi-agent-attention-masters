"""On-policy rollout buffer for MAT.

Same GAE/PPO role as ``agents/mappo/buffer.py``, but shaped for a *sequence*
model: every array carries an explicit agent axis, ``(T, n_agent, ...)``, since
MAT's shared transformer consumes all agents in one forward pass. Consequently
minibatches are drawn over timesteps and always contain every agent together --
there is no per-agent generator like MAPPO's, because agents cannot be scored
independently once the decoder conditions each on the others.

Observations are stored pre-padded to the common ``obs_dim`` (see
``agents/mat/mat.py``), so a single rectangular array covers heterogeneous envs.
"""

from __future__ import annotations

import numpy as np
import torch


class MATRolloutBuffer:
    def __init__(
        self,
        max_steps: int,
        obs_dim: int,
        act_dim: int,
        num_agents: int,
        device: torch.device,
        is_discrete: bool = True,
    ):
        self.max_steps = max_steps
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.device = device
        self.is_discrete = is_discrete

        # Discrete stores one index per agent; continuous stores the full vector.
        act_store = 1 if is_discrete else act_dim

        self.obs_buf = np.zeros((max_steps, num_agents, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_steps, num_agents, act_store), dtype=np.float32)
        self.logprob_buf = np.zeros((max_steps, num_agents, 1), dtype=np.float32)
        self.val_buf = np.zeros((max_steps, num_agents, 1), dtype=np.float32)
        self.rew_buf = np.zeros((max_steps, num_agents, 1), dtype=np.float32)
        self.mask_buf = np.zeros((max_steps, num_agents, 1), dtype=np.float32)  # 1 - terminated
        self.ret_buf = np.zeros((max_steps, num_agents, 1), dtype=np.float32)
        self.adv_buf = np.zeros((max_steps, num_agents, 1), dtype=np.float32)

        self.step = 0

    def reset(self) -> None:
        self.step = 0

    def insert(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        values: np.ndarray,
        rewards: np.ndarray,
        masks: np.ndarray,
    ) -> None:
        """Store one timestep. All arrays are agent-major: ``(n_agent, ...)``."""
        t = self.step
        self.obs_buf[t] = obs
        self.act_buf[t] = actions
        self.logprob_buf[t] = np.asarray(log_probs, dtype=np.float32).reshape(self.num_agents, 1)
        self.val_buf[t] = np.asarray(values, dtype=np.float32).reshape(self.num_agents, 1)
        self.rew_buf[t] = np.asarray(rewards, dtype=np.float32).reshape(self.num_agents, 1)
        self.mask_buf[t] = np.asarray(masks, dtype=np.float32).reshape(self.num_agents, 1)
        self.step += 1

    def compute_returns(self, next_value: np.ndarray, gamma: float, gae_lambda: float) -> None:
        """Per-agent GAE over the rollout (same recursion as MAPPO's buffer).

        ``mask`` (1 - terminated) cuts the bootstrap on true terminations while
        preserving it on time-limit truncations.
        """
        T = self.step
        next_value = np.asarray(next_value, dtype=np.float32).reshape(self.num_agents, 1)
        for i in range(self.num_agents):
            gae = 0.0
            for t in reversed(range(T)):
                next_val = next_value[i, 0] if t == T - 1 else self.val_buf[t + 1, i, 0]
                mask = self.mask_buf[t, i, 0]
                delta = self.rew_buf[t, i, 0] + gamma * next_val * mask - self.val_buf[t, i, 0]
                gae = delta + gamma * gae_lambda * mask * gae
                self.adv_buf[t, i, 0] = gae
                self.ret_buf[t, i, 0] = gae + self.val_buf[t, i, 0]

    def minibatch_generator(self, num_mini_batch: int):
        """Yield shuffled minibatches of whole timesteps (all agents together).

        Advantages are normalized once across the entire rollout (over agents and
        timesteps jointly), matching the reference trainer, which normalizes the
        full advantage array rather than per agent.
        """
        T = self.step
        adv = self.adv_buf[:T].copy()
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        indices = np.random.permutation(T)
        mb_size = max(1, T // max(1, num_mini_batch))
        for start in range(0, T, mb_size):
            mb = indices[start:start + mb_size]
            yield {
                "obs": torch.as_tensor(self.obs_buf[:T][mb], device=self.device),
                "actions": torch.as_tensor(self.act_buf[:T][mb], device=self.device),
                "old_log_probs": torch.as_tensor(self.logprob_buf[:T][mb], device=self.device),
                "returns": torch.as_tensor(self.ret_buf[:T][mb], device=self.device),
                "advantages": torch.as_tensor(adv[mb], device=self.device),
                "values": torch.as_tensor(self.val_buf[:T][mb], device=self.device),
            }

    def __len__(self) -> int:
        return self.step
