"""Autoregressive vs. parallel action computation for MAT.

MAT's defining mechanism: actions are generated *sequentially over agents*.
Agent ``i``'s action distribution is conditioned on agents ``0..i-1``'s already
chosen actions, which is what gives the multi-agent advantage decomposition its
monotonic-improvement guarantee (Wen et al., 2022).

Two paths compute the same distributions:

* ``*_autoregressive_act`` -- rollout time. One decoder pass per agent, feeding
  each sampled action into the next agent's input. O(n_agent) passes.
* ``*_parallel_act`` -- update time. Stored actions are already known, so they
  are shifted right by one agent and the whole sequence is scored in a *single*
  pass; the decoder's causal mask makes this equivalent to the loop above.

``action_mask`` (batch, n_agent, action_dim) marks valid actions per agent with
1. It is how heterogeneous action spaces are supported: agents whose action set
is smaller than the padded ``action_dim`` get their surplus logits driven to
-inf so they can never be sampled.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical, Normal
from torch.nn import functional as F

# Large negative constant used to zero out invalid logits after softmax.
_NEG_INF = -1e10


def _masked_categorical(logit: torch.Tensor, action_mask: torch.Tensor | None) -> Categorical:
    if action_mask is not None:
        logit = logit.masked_fill(action_mask == 0, _NEG_INF)
    return Categorical(logits=logit)


def discrete_autoregressive_act(
    decoder,
    obs_rep: torch.Tensor,
    batch_size: int,
    n_agent: int,
    action_dim: int,
    device: torch.device,
    action_mask: torch.Tensor | None = None,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one action per agent, sequentially. Returns ``(actions, log_probs)``."""
    # Slot 0 is the "start" token; slots 1: hold the previous agent's one-hot.
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1), dtype=torch.float32, device=device)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long, device=device)
    output_action_log = torch.zeros((batch_size, n_agent, 1), dtype=torch.float32, device=device)

    for i in range(n_agent):
        logit = decoder(shifted_action, obs_rep)[:, i, :]
        mask_i = action_mask[:, i, :] if action_mask is not None else None
        distri = _masked_categorical(logit, mask_i)

        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, 0] = action
        output_action_log[:, i, 0] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim).float()

    return output_action, output_action_log


def discrete_parallel_act(
    decoder,
    obs_rep: torch.Tensor,
    action: torch.Tensor,
    batch_size: int,
    n_agent: int,
    action_dim: int,
    device: torch.device,
    action_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score stored ``action`` (batch, n_agent, 1) in one pass. ``(log_probs, entropy)``."""
    one_hot_action = F.one_hot(action.squeeze(-1).long(), num_classes=action_dim).float()
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1), dtype=torch.float32, device=device)
    shifted_action[:, 0, 0] = 1
    shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]

    logit = decoder(shifted_action, obs_rep)
    distri = _masked_categorical(logit, action_mask)

    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    return action_log, entropy


def continuous_autoregressive_act(
    decoder,
    obs_rep: torch.Tensor,
    batch_size: int,
    n_agent: int,
    action_dim: int,
    device: torch.device,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous counterpart: the previous agent's action vector is fed forward."""
    shifted_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32, device=device)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32, device=device)
    output_action_log = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32, device=device)

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep)[:, i, :]
        # Reference parameterization: bounded std via sigmoid, not exp(log_std).
        action_std = torch.sigmoid(decoder.log_std) * 0.5
        distri = Normal(act_mean, action_std)

        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

    return output_action, output_action_log


def continuous_parallel_act(
    decoder,
    obs_rep: torch.Tensor,
    action: torch.Tensor,
    batch_size: int,
    n_agent: int,
    action_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    shifted_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32, device=device)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy
