"""Transformer encoder/decoder for MAT (Multi-Agent Transformer).

Ported from the reference implementation in ``agents/Multi-Agent-Transformer``
(``mat/algorithms/mat/algorithm/ma_transformer.py``), with the agent dimension
treated as the sequence dimension: one *shared* model processes all agents at
once, in contrast to this repo's MAAC/MADDPG/MAPPO which instantiate per-agent
networks.

Two deliberate departures from upstream:

* Upstream sizes every agent from ``observation_space[0]`` and so requires all
  agents to share obs/action dims. Only ``simple_spread_v3`` satisfies that
  among this repo's MPE2 envs, so here observations are zero-padded to
  ``obs_dim = max_i obs_dim_i`` and an ``action_mask`` invalidates the padded
  logits of agents with smaller action sets. That keeps the shared-sequence
  architecture while covering the heterogeneous envs too.
* Upstream carries a ``state_dim``/``encode_state`` path that its own code
  disables (it overwrites the state with ``np.zeros((..., 37))``). That dead
  path is dropped; the encoder always runs on observations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


def init_(module: nn.Linear, gain: float = 0.01, activate: bool = False) -> nn.Linear:
    """Orthogonal init with zero bias, matching the reference implementation."""
    if activate:
        gain = nn.init.calculate_gain("relu")
    nn.init.orthogonal_(module.weight, gain=gain)
    # The discrete action encoder is built with bias=False (as upstream).
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
    return module


class SelfAttention(nn.Module):
    """Multi-head attention over the agent sequence.

    ``masked=True`` applies a causal (lower-triangular) mask so agent ``i`` only
    attends to agents ``<= i`` -- this is what makes the decoder's action
    generation autoregressive over agents.
    """

    def __init__(self, n_embd: int, n_head: int, n_agent: int, masked: bool = False):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must be divisible by n_head"
        self.masked = masked
        self.n_head = n_head

        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        self.proj = init_(nn.Linear(n_embd, n_embd))

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(n_agent + 1, n_agent + 1)).view(1, 1, n_agent + 1, n_agent + 1),
        )

    def forward(self, key: torch.Tensor, value: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        B, L, D = query.size()
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        return self.proj(y)


class EncodeBlock(nn.Module):
    """Unmasked transformer block: every agent attends to every other agent."""

    def __init__(self, n_embd: int, n_head: int, n_agent: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_embd, n_embd)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class DecodeBlock(nn.Module):
    """Masked self-attention over actions, then cross-attention to the encoding."""

    def __init__(self, n_embd: int, n_head: int, n_agent: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(n_embd, n_embd)),
        )

    def forward(self, x: torch.Tensor, obs_rep: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(obs_rep + self.attn2(key=x, value=x, query=obs_rep))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):
    """Encodes all agents' observations jointly and emits per-agent values.

    The value head here *is* MAT's critic: because the blocks are unmasked, each
    agent's value is conditioned on every agent's observation, which is what
    makes it centralized in the CTDE sense.
    """

    def __init__(self, obs_dim: int, n_block: int, n_embd: int, n_head: int, n_agent: int):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            init_(nn.Linear(obs_dim, n_embd), activate=True),
            nn.GELU(),
        )
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, 1)),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``obs``: (batch, n_agent, obs_dim) -> ``(values, obs_rep)``."""
        x = self.obs_encoder(obs)
        rep = self.blocks(self.ln(x))
        values = self.head(rep)
        return values, rep


class Decoder(nn.Module):
    """Maps (shifted actions, observation encoding) to per-agent action logits.

    Discrete actions are fed in as one-hot vectors with an extra leading "start"
    slot, so agent 0 conditions on a start token and agent ``i`` on agents
    ``< i``'s actions.
    """

    def __init__(
        self,
        action_dim: int,
        n_block: int,
        n_embd: int,
        n_head: int,
        n_agent: int,
        is_discrete: bool = True,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.is_discrete = is_discrete

        if not is_discrete:
            self.log_std = nn.Parameter(torch.ones(action_dim))

        in_dim = action_dim + 1 if is_discrete else action_dim
        self.action_encoder = nn.Sequential(
            init_(nn.Linear(in_dim, n_embd, bias=not is_discrete), activate=True),
            nn.GELU(),
        )
        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.ModuleList([DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, action_dim)),
        )

    def forward(self, action: torch.Tensor, obs_rep: torch.Tensor) -> torch.Tensor:
        x = self.ln(self.action_encoder(action))
        for block in self.blocks:
            x = block(x, obs_rep)
        return self.head(x)
