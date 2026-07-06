import torch
import torch.nn as nn
from torch.optim import Adam

from agents.maac_continuous.utils.agents import AttentionAgentContinuous
from agents.maac_continuous.utils.critics import AttentionCriticContinuous
from agents.maac_continuous.utils.misc import (
    disable_gradients,
    enable_gradients,
    hard_update,
    soft_update,
)

MSELoss = nn.MSELoss()


class AttentionSACContinuous:
    """
    MAAC adapted for continuous action spaces.

    Policy: Gaussian with tanh squashing + reparameterization trick.
    Critic: Attention-based, scalar Q per agent. The Q head receives the agent's
            own SA encoding so the reparameterization gradient flows through it.
    Update: SAC-style — entropy-regularized Bellman target for critic,
            reparameterization policy gradient per agent.
    """

    def __init__(self, agent_init_params, sa_size,
                 gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                 reward_scale=10.0, pol_hidden_dim=128, critic_hidden_dim=128,
                 attend_heads=4, **kwargs):
        self.nagents = len(sa_size)
        self.gamma = gamma
        self.tau = tau
        self.pi_lr = pi_lr
        self.q_lr = q_lr
        self.reward_scale = reward_scale

        self.agents = [
            AttentionAgentContinuous(lr=pi_lr, hidden_dim=pol_hidden_dim, **params)
            for params in agent_init_params
        ]
        self.critic = AttentionCriticContinuous(
            sa_size, hidden_dim=critic_hidden_dim, attend_heads=attend_heads
        )
        self.target_critic = AttentionCriticContinuous(
            sa_size, hidden_dim=critic_hidden_dim, attend_heads=attend_heads
        )
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=q_lr, weight_decay=1e-3)

        self.pol_dev = "cpu"
        self.critic_dev = "cpu"
        self.trgt_pol_dev = "cpu"
        self.trgt_critic_dev = "cpu"
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def step(self, observations, explore=True):
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents, observations)]

    def update_critic(self, sample, soft=True, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones = sample

        with torch.no_grad():
            next_acs, next_log_pis = [], []
            for pi, nobs in zip(self.target_policies, next_obs):
                na, nlp = pi(nobs, sample=True)
                next_acs.append(na)
                next_log_pis.append(nlp)
            next_qs = self.target_critic(list(zip(next_obs, next_acs)))

        critic_rets = self.critic(list(zip(obs, acs)), regularize=True,
                                  logger=logger, niter=self.niter)

        q_loss = 0
        for a_i, nq, log_pi, (pq, regs) in zip(
            range(self.nagents), next_qs, next_log_pis, critic_rets
        ):
            target_q = rews[a_i].view(-1, 1) + self.gamma * nq * (1 - dones[a_i].view(-1, 1))
            if soft:
                target_q = target_q - log_pi / self.reward_scale
            q_loss += MSELoss(pq, target_q.detach())
            for reg in regs:
                q_loss += reg

        q_loss.backward()
        self.critic.scale_shared_grads()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), 10 * self.nagents
        )
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()

        if logger is not None:
            logger.add_scalar("losses/q_loss", q_loss.item(), self.niter)
            logger.add_scalar("grad_norms/q", grad_norm, self.niter)
        self.niter += 1

    def update_policies(self, sample, soft=True, logger=None, **kwargs):
        obs, acs, rews, next_obs, dones = sample

        for a_i in range(self.nagents):
            curr_ac, log_pi = self.agents[a_i].policy(obs[a_i], sample=True)

            # Per-agent critic call: agent a_i uses its reparameterized action
            # (gradient flows through); others use buffer actions (no cross-agent
            # gradient — only a_i's policy parameters are updated this step).
            critic_in = [
                (obs[j], curr_ac if j == a_i else acs[j])
                for j in range(self.nagents)
            ]

            disable_gradients(self.critic)
            critic_rets = self.critic(critic_in)
            enable_gradients(self.critic)

            q_i = critic_rets[a_i]  # [batch, 1]

            if soft:
                pol_loss = (log_pi / self.reward_scale - q_i).mean()
            else:
                pol_loss = -q_i.mean()

            pol_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agents[a_i].policy.parameters(), 0.5
            )
            self.agents[a_i].policy_optimizer.step()
            self.agents[a_i].policy_optimizer.zero_grad()

            if logger is not None:
                logger.add_scalar(f"agent{a_i}/losses/pol_loss", pol_loss.item(), self.niter)
                logger.add_scalar(f"agent{a_i}/grad_norms/pi", grad_norm, self.niter)
                logger.add_scalar(f"agent{a_i}/policy_entropy", -log_pi.mean().item(), self.niter)

    def update_all_targets(self):
        soft_update(self.target_critic, self.critic, self.tau)
        for a in self.agents:
            soft_update(a.target_policy, a.policy, self.tau)

    def prep_training(self, device="gpu"):
        self.critic.train()
        self.target_critic.train()
        for a in self.agents:
            a.policy.train()
            a.target_policy.train()
        fn = (lambda x: x.cuda()) if device == "gpu" else (lambda x: x.cpu())
        if self.pol_dev != device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if self.critic_dev != device:
            self.critic = fn(self.critic)
            self.critic_dev = device
        if self.trgt_pol_dev != device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if self.trgt_critic_dev != device:
            self.target_critic = fn(self.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device="cpu"):
        for a in self.agents:
            a.policy.eval()
        fn = (lambda x: x.cuda()) if device == "gpu" else (lambda x: x.cpu())
        if self.pol_dev != device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        self.prep_training(device="cpu")
        torch.save(
            {
                "init_dict": self.init_dict,
                "agent_params": [a.get_params() for a in self.agents],
                "critic_params": {
                    "critic": self.critic.state_dict(),
                    "target_critic": self.target_critic.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                },
            },
            filename,
        )

    @classmethod
    def init_from_env(cls, env, gamma=0.95, tau=0.01, pi_lr=0.01, q_lr=0.01,
                      reward_scale=10.0, pol_hidden_dim=128, critic_hidden_dim=128,
                      attend_heads=4, **kwargs):
        agent_ids = list(env.agents)
        agent_init_params, sa_size = [], []
        for agent_id in agent_ids:
            obsp = env.observation_space(agent_id)
            acsp = env.action_space(agent_id)
            agent_init_params.append({
                "obs_dim": obsp.shape[0],
                "act_dim": acsp.shape[0],
                "act_low": acsp.low,
                "act_high": acsp.high,
            })
            sa_size.append((obsp.shape[0], acsp.shape[0]))

        init_dict = dict(
            gamma=gamma, tau=tau, pi_lr=pi_lr, q_lr=q_lr,
            reward_scale=reward_scale, pol_hidden_dim=pol_hidden_dim,
            critic_hidden_dim=critic_hidden_dim, attend_heads=attend_heads,
            agent_init_params=agent_init_params, sa_size=sa_size,
        )
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, load_critic=False):
        save_dict = torch.load(filename, weights_only=False)
        instance = cls(**save_dict["init_dict"])
        instance.init_dict = save_dict["init_dict"]
        for a, params in zip(instance.agents, save_dict["agent_params"]):
            a.load_params(params)
        if load_critic:
            cp = save_dict["critic_params"]
            instance.critic.load_state_dict(cp["critic"])
            instance.target_critic.load_state_dict(cp["target_critic"])
            instance.critic_optimizer.load_state_dict(cp["critic_optimizer"])
        return instance
