import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN = -5
LOG_STD_MAX = 2


class GaussianPolicy(nn.Module):
    """
    Gaussian policy with tanh-squashed actions for continuous action spaces.
    Uses the reparameterization trick: action = tanh(mean + std * noise).
    """

    def __init__(self, obs_dim, act_dim, hidden_dim=64, norm_in=True,
                 act_low=None, act_high=None):
        super().__init__()
        self.in_fn = nn.BatchNorm1d(obs_dim, affine=False) if norm_in else nn.Identity()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

        if act_low is not None and act_high is not None:
            lo = torch.as_tensor(act_low, dtype=torch.float32)
            hi = torch.as_tensor(act_high, dtype=torch.float32)
            self.register_buffer("act_scale", (hi - lo) / 2.0)
            self.register_buffer("act_bias", (hi + lo) / 2.0)
        else:
            self.register_buffer("act_scale", torch.ones(act_dim))
            self.register_buffer("act_bias", torch.zeros(act_dim))

    def forward(self, obs, sample=True):
        """
        Returns:
            action: scaled continuous action, shape [batch, act_dim]
            log_pi: log probability including tanh and scaling Jacobians, shape [batch, 1]
        """
        x = F.leaky_relu(self.fc1(self.in_fn(obs)))
        x = F.leaky_relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        raw = mean + std * torch.randn_like(mean) if sample else mean
        tanh_raw = torch.tanh(raw)
        action = tanh_raw * self.act_scale + self.act_bias

        # Gaussian log prob + tanh Jacobian + linear scale Jacobian
        log_pi = (
            -0.5 * ((raw - mean) / (std + 1e-6)).pow(2)
            - log_std
            - 0.5 * np.log(2 * np.pi)
        ).sum(dim=1, keepdim=True)
        log_pi -= torch.log(1.0 - tanh_raw.pow(2) + 1e-6).sum(dim=1, keepdim=True)
        log_pi -= torch.log(self.act_scale + 1e-6).sum()

        return action, log_pi
