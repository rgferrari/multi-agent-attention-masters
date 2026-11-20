"""PyTorch MAT (Multi-Agent Transformer) implementation for parallel envs.

MAT (Wen et al., 2022, "Multi-Agent Reinforcement Learning is a Sequence
Modeling Problem") recasts MARL as sequence modeling: a single encoder-decoder
transformer takes all agents' observations as a sequence and emits actions
*autoregressively over agents*, so agent ``i``'s action is conditioned on agents
``0..i-1``'s actions. Trained with PPO, like this repo's MAPPO.

How it differs from the other agents here:

* **One shared model, not per-agent networks.** MAAC/MADDPG/MAPPO all build a
  separate actor (and critic) per agent; MAT has a single transformer whose
  agent axis is the sequence axis. Adding agents changes sequence length, not
  parameter count.
* **The critic is the encoder's value head.** There is no separate critic
  network: the unmasked encoder blocks let each agent's value attend to every
  agent's observation, which is what makes it centralized (CTDE).
* **Actions are sequentially dependent.** MAPPO's agents choose independently
  given their observations; MAT's agent ``i`` sees what ``0..i-1`` chose.

Heterogeneous observation/action spaces are handled by padding to the per-env
maximum and masking invalid actions -- see ``agents/mat/models.py``. Agent order
is the env's ``agent_ids`` order, which fixes the autoregressive ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.mat.models import Decoder, Encoder
from agents.mat.transformer_act import (
    continuous_autoregressive_act,
    continuous_parallel_act,
    discrete_autoregressive_act,
    discrete_parallel_act,
)


@dataclass
class MATConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    ppo_epoch: int = 15
    num_mini_batch: int = 1
    entropy_coef: float = 0.01
    value_loss_coef: float = 1.0
    max_grad_norm: float = 10.0
    lr: float = 5e-4
    n_block: int = 1
    n_embd: int = 64
    n_head: int = 1
    use_clipped_value_loss: bool = True


class MAT:
    def __init__(
        self,
        agent_ids: Sequence[str],
        obs_spaces: Sequence,
        act_spaces: Sequence,
        device: torch.device,
        config: MATConfig | None = None,
    ) -> None:
        self.agent_ids = list(agent_ids)
        self.device = device
        self.config = config or MATConfig()
        self.num_agents = len(self.agent_ids)

        # --- Space handling -------------------------------------------------
        # The shared transformer needs one obs_dim / action_dim for the whole
        # sequence, so pad to the max and remember each agent's true size.
        self.true_obs_dims = [int(space.shape[0]) for space in obs_spaces]
        self.obs_dim = max(self.true_obs_dims)

        first = act_spaces[0]
        self.is_discrete = hasattr(first, "n")
        if self.is_discrete:
            if any(hasattr(s, "n") != self.is_discrete for s in act_spaces):
                raise ValueError("MAT requires all agents to share one action-space type")
            self.true_act_dims = [int(s.n) for s in act_spaces]
            self.action_dim = max(self.true_act_dims)
            self.act_low = None
            self.act_high = None
        else:
            self.true_act_dims = [int(s.shape[0]) for s in act_spaces]
            self.action_dim = max(self.true_act_dims)
            self.act_low = [np.asarray(s.low, dtype=np.float32) for s in act_spaces]
            self.act_high = [np.asarray(s.high, dtype=np.float32) for s in act_spaces]

        # Per-agent validity mask over the padded action axis. Only meaningful
        # for discrete spaces, where surplus logits must be made unsamplable.
        if self.is_discrete:
            mask = np.zeros((self.num_agents, self.action_dim), dtype=np.float32)
            for i, d in enumerate(self.true_act_dims):
                mask[i, :d] = 1.0
            self.action_mask = torch.as_tensor(mask, device=device).unsqueeze(0)
        else:
            self.action_mask = None

        self.heterogeneous = len(set(self.true_obs_dims)) > 1 or len(set(self.true_act_dims)) > 1

        # --- Networks -------------------------------------------------------
        cfg = self.config
        self.encoder = Encoder(self.obs_dim, cfg.n_block, cfg.n_embd, cfg.n_head, self.num_agents).to(device)
        self.decoder = Decoder(
            self.action_dim, cfg.n_block, cfg.n_embd, cfg.n_head, self.num_agents, is_discrete=self.is_discrete
        ).to(device)

        # One optimizer over the whole model: encoder and decoder are trained
        # jointly by a single combined loss (unlike MAPPO's split actor/critic).
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=cfg.lr
        )

    # ------------------------------------------------------------------ utils
    def _stack_obs(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Dict of per-agent obs -> zero-padded ``(n_agent, obs_dim)`` array."""
        out = np.zeros((self.num_agents, self.obs_dim), dtype=np.float32)
        for i, agent_id in enumerate(self.agent_ids):
            o = np.asarray(obs[agent_id], dtype=np.float32).ravel()
            out[i, : o.shape[0]] = o
        return out

    def _obs_tensor(self, obs: Dict[str, np.ndarray]) -> torch.Tensor:
        return torch.as_tensor(self._stack_obs(obs), device=self.device).unsqueeze(0)

    def train(self) -> None:
        self.encoder.train()
        self.decoder.train()

    def eval(self) -> None:
        self.encoder.eval()
        self.decoder.eval()

    # -------------------------------------------------------------- rollout
    def select_actions(
        self, obs: Dict[str, np.ndarray], deterministic: bool = False
    ) -> Tuple[Dict[str, int | np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Roll out one step for every agent autoregressively.

        Returns ``(actions_env, actions_store, log_probs, values)`` mirroring
        ``MAPPO.select_actions``, except the latter three are agent-major arrays
        rather than lists, since the buffer stores them stacked.
        """
        obs_t = self._obs_tensor(obs)
        self.eval()
        with torch.no_grad():
            values, obs_rep = self.encoder(obs_t)
            if self.is_discrete:
                actions, log_probs = discrete_autoregressive_act(
                    self.decoder, obs_rep, 1, self.num_agents, self.action_dim,
                    self.device, self.action_mask, deterministic,
                )
            else:
                actions, log_probs = continuous_autoregressive_act(
                    self.decoder, obs_rep, 1, self.num_agents, self.action_dim,
                    self.device, deterministic,
                )

        actions_np = actions.squeeze(0).cpu().numpy()
        log_probs_np = log_probs.squeeze(0).cpu().numpy()
        values_np = values.squeeze(0).cpu().numpy()

        actions_env: Dict[str, int | np.ndarray] = {}
        for i, agent_id in enumerate(self.agent_ids):
            if self.is_discrete:
                actions_env[agent_id] = int(actions_np[i, 0])
            else:
                # Trim padding back to this agent's real action width, then clip.
                d = self.true_act_dims[i]
                vec = actions_np[i, :d].astype(np.float32)
                actions_env[agent_id] = np.clip(vec, self.act_low[i], self.act_high[i])

        # Continuous log_probs are per-dimension; sum to one scalar per agent so
        # the PPO ratio is over the joint action, matching MAPPO's convention.
        if not self.is_discrete:
            log_probs_np = log_probs_np.sum(axis=-1, keepdims=True)

        return actions_env, actions_np, log_probs_np, values_np

    def get_values(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        """Bootstrap V(s) per agent from the (final) observation."""
        obs_t = self._obs_tensor(obs)
        self.eval()
        with torch.no_grad():
            values, _ = self.encoder(obs_t)
        return values.squeeze(0).cpu().numpy()

    # --------------------------------------------------------------- update
    def update(self, buffer) -> Dict[str, float]:
        """PPO update over the shared transformer.

        Unlike MAPPO -- which loops over agents and steps a separate optimizer
        each -- this is a single backward pass over one joint loss, because the
        encoder and decoder are shared across all agents.
        """
        cfg = self.config
        self.train()

        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropies: List[float] = []
        ratios: List[float] = []

        for _ in range(cfg.ppo_epoch):
            for mb in buffer.minibatch_generator(cfg.num_mini_batch):
                obs = mb["obs"]
                actions = mb["actions"]
                batch_size = obs.shape[0]

                mask = None
                if self.action_mask is not None:
                    mask = self.action_mask.expand(batch_size, -1, -1)

                values, obs_rep = self.encoder(obs)
                if self.is_discrete:
                    log_probs, entropy = discrete_parallel_act(
                        self.decoder, obs_rep, actions, batch_size,
                        self.num_agents, self.action_dim, self.device, mask,
                    )
                else:
                    log_probs, entropy = continuous_parallel_act(
                        self.decoder, obs_rep, actions, batch_size,
                        self.num_agents, self.action_dim, self.device,
                    )
                    # Match the summed rollout log_probs (joint over dims).
                    log_probs = log_probs.sum(dim=-1, keepdim=True)
                    entropy = entropy.sum(dim=-1, keepdim=True)

                advantages = mb["advantages"]
                ratio = torch.exp(log_probs - mb["old_log_probs"])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_param, 1.0 + cfg.clip_param) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                returns = mb["returns"]
                if cfg.use_clipped_value_loss:
                    old_values = mb["values"]
                    v_clipped = old_values + torch.clamp(values - old_values, -cfg.clip_param, cfg.clip_param)
                    value_loss = 0.5 * torch.max((values - returns) ** 2, (v_clipped - returns) ** 2).mean()
                else:
                    value_loss = 0.5 * ((values - returns) ** 2).mean()

                entropy_mean = entropy.mean()
                loss = policy_loss - cfg.entropy_coef * entropy_mean + cfg.value_loss_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()), cfg.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_mean.item()))
                ratios.append(float(ratio.mean().item()))

        n = max(1, len(policy_losses))
        return {
            "policy_loss": sum(policy_losses) / n,
            "value_loss": sum(value_losses) / n,
            "entropy": sum(entropies) / n,
            "ratio": sum(ratios) / n,
        }

    # ----------------------------------------------------------- checkpoints
    def state_dict(self) -> Dict[str, object]:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.__dict__,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        if "encoder" in state:
            self.encoder.load_state_dict(state["encoder"])
        if "decoder" in state:
            self.decoder.load_state_dict(state["decoder"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
