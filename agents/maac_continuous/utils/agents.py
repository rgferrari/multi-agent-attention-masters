from torch.optim import Adam
from agents.maac_continuous.utils.misc import hard_update
from agents.maac_continuous.utils.policies import GaussianPolicy


class AttentionAgentContinuous:
    def __init__(self, obs_dim, act_dim, hidden_dim=64, lr=0.01,
                 act_low=None, act_high=None):
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dim,
                                     act_low=act_low, act_high=act_high)
        self.target_policy = GaussianPolicy(obs_dim, act_dim, hidden_dim,
                                            act_low=act_low, act_high=act_high)
        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)

    def step(self, obs, explore=True):
        return self.policy(obs, sample=explore)

    def get_params(self):
        return {
            "policy": self.policy.state_dict(),
            "target_policy": self.target_policy.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
        }

    def load_params(self, params):
        self.policy.load_state_dict(params["policy"])
        self.target_policy.load_state_dict(params["target_policy"])
        self.policy_optimizer.load_state_dict(params["policy_optimizer"])
