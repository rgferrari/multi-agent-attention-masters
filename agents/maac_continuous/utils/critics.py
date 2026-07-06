import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain


class AttentionCriticContinuous(nn.Module):
    """
    Attention critic for continuous action spaces.

    Compared to the discrete AttentionCritic:
    - Outputs a scalar Q instead of Q-per-action.
    - The Q head receives the agent's own SA encoding (state+action) rather than
      the state-only encoding, so Q depends on the agent's own continuous action.
      This enables the reparameterization gradient to flow through the critic.
    - Selectors (attention queries) still come from the state-only encoder,
      preserving the original attention design.
    """

    def __init__(self, sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1):
        super().__init__()
        assert hidden_dim % attend_heads == 0
        self.sa_sizes = sa_sizes
        self.nagents = len(sa_sizes)
        self.attend_heads = attend_heads

        self.critic_encoders = nn.ModuleList()
        self.critics = nn.ModuleList()
        self.state_encoders = nn.ModuleList()

        for sdim, adim in sa_sizes:
            encoder = nn.Sequential()
            if norm_in:
                encoder.add_module("enc_bn", nn.BatchNorm1d(sdim + adim, affine=False))
            encoder.add_module("enc_fc1", nn.Linear(sdim + adim, hidden_dim))
            encoder.add_module("enc_nl", nn.LeakyReLU())
            self.critic_encoders.append(encoder)

            # Q head: sa_encoding_i (hidden_dim) + attended values from others (hidden_dim)
            critic = nn.Sequential()
            critic.add_module("critic_fc1", nn.Linear(2 * hidden_dim, hidden_dim))
            critic.add_module("critic_nl", nn.LeakyReLU())
            critic.add_module("critic_fc2", nn.Linear(hidden_dim, 1))
            self.critics.append(critic)

            state_encoder = nn.Sequential()
            if norm_in:
                state_encoder.add_module("s_enc_bn", nn.BatchNorm1d(sdim, affine=False))
            state_encoder.add_module("s_enc_fc1", nn.Linear(sdim, hidden_dim))
            state_encoder.add_module("s_enc_nl", nn.LeakyReLU())
            self.state_encoders.append(state_encoder)

        attend_dim = hidden_dim // attend_heads
        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()
        self.value_extractors = nn.ModuleList()
        for _ in range(attend_heads):
            self.key_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.selector_extractors.append(nn.Linear(hidden_dim, attend_dim, bias=False))
            self.value_extractors.append(
                nn.Sequential(nn.Linear(hidden_dim, attend_dim), nn.LeakyReLU())
            )

        self.shared_modules = [
            self.key_extractors, self.selector_extractors,
            self.value_extractors, self.critic_encoders,
        ]

    def shared_parameters(self):
        return chain(*[m.parameters() for m in self.shared_modules])

    def scale_shared_grads(self):
        for p in self.shared_parameters():
            p.grad.data.mul_(1.0 / self.nagents)

    def forward(self, inps, regularize=False, return_attend=False, logger=None, niter=0):
        """
        Args:
            inps: list of (obs, action) tuples, one per agent
        Returns:
            list of scalar Q tensors [batch, 1], one per agent
        """
        states = [s for s, a in inps]
        inps_cat = [torch.cat((s, a), dim=1) for s, a in inps]

        sa_encodings = [enc(x) for enc, x in zip(self.critic_encoders, inps_cat)]
        s_encodings = [self.state_encoders[i](states[i]) for i in range(self.nagents)]

        all_head_keys = [[k_ext(enc) for enc in sa_encodings] for k_ext in self.key_extractors]
        all_head_values = [[v_ext(enc) for enc in sa_encodings] for v_ext in self.value_extractors]
        all_head_selectors = [[sel_ext(enc) for enc in s_encodings] for sel_ext in self.selector_extractors]

        # For each agent: attend over all other agents' SA encodings
        other_all_values = [[] for _ in range(self.nagents)]
        all_attend_logits = [[] for _ in range(self.nagents)]
        all_attend_probs = [[] for _ in range(self.nagents)]

        for head_keys, head_values, head_selectors in zip(
            all_head_keys, all_head_values, all_head_selectors
        ):
            for a_i in range(self.nagents):
                keys = [k for j, k in enumerate(head_keys) if j != a_i]
                values = [v for j, v in enumerate(head_values) if j != a_i]
                selector = head_selectors[a_i]

                attend_logits = torch.matmul(
                    selector.view(selector.shape[0], 1, -1),
                    torch.stack(keys).permute(1, 2, 0),
                )
                scaled = attend_logits / np.sqrt(keys[0].shape[1])
                weights = F.softmax(scaled, dim=2)
                attended = (torch.stack(values).permute(1, 2, 0) * weights).sum(dim=2)

                other_all_values[a_i].append(attended)
                all_attend_logits[a_i].append(attend_logits)
                all_attend_probs[a_i].append(weights)

        if return_attend:
            # Per agent: [n_heads, batch, 1, n_other] of softmax attention weights.
            return [
                np.stack([p.detach().cpu().numpy() for p in all_attend_probs[a_i]])
                for a_i in range(self.nagents)
            ]

        all_qs = []
        for a_i in range(self.nagents):
            head_entropies = [
                (-(probs + 1e-8).log() * probs).sum(dim=2).mean()
                for probs in all_attend_probs[a_i]
            ]
            # Q depends on own SA encoding + attended context from others
            critic_in = torch.cat((sa_encodings[a_i], *other_all_values[a_i]), dim=1)
            q = self.critics[a_i](critic_in)  # [batch, 1]

            if regularize:
                attend_mag_reg = 1e-3 * sum(
                    (logit**2).mean() for logit in all_attend_logits[a_i]
                )
                all_qs.append((q, (attend_mag_reg,)))
            else:
                all_qs.append(q)

            if logger is not None:
                logger.add_scalar(
                    f"agent{a_i}/attention/mean_entropy",
                    torch.stack(head_entropies).mean(),
                    niter,
                )

        return all_qs
