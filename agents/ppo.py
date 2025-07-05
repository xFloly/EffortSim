# agents/ppo.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.distributions import Normal


# ------------------------------------------------------------
#  Sieci
# ------------------------------------------------------------
class Actor(nn.Module):
    """Prosty MLP → (mean, std) dla ciągłych akcji z tanh-squashem."""
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)            # mean
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        mean = self.net(x)
        std  = self.log_std.exp()
        return mean, std


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # shape (B,)


# ------------------------------------------------------------
#  PPO Agent
# ------------------------------------------------------------
class PPOAgent:
    def __init__(self, agent_id, obs_dim, action_dim, device, cfg):
        self.agent_id = agent_id
        self.device   = device
        self.cfg      = cfg

        self.actor  = Actor(obs_dim, action_dim).to(device)
        self.critic = Critic(obs_dim).to(device)

        # self.model = self.actor 

        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=cfg.learning_rate
        )

        # Recurrent experience buffers
        self.obs_buf   = []
        self.act_buf   = []   # squashed action (tanh)
        self.u_buf     = []   # unsquashed action before tanh
        self.logp_buf  = []
        self.rew_buf   = []
        self.done_buf  = []
        self.val_buf   = []

        self.steps_done = 0
        self.episode    = 0

    # --------------------------------------------------------
    #  Generowanie akcji
    # --------------------------------------------------------
    def act(self, obs):
        """Zwraca:
           - action (numpy, już po tanh, do środowiska)
           - log_prob(action)  (scalar Python float)
           - V(s)              (scalar Python float)
           - u (numpy, przed tanh)  ← trzeba zapisać w buforze!
        """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            mean, std = self.actor(obs_tensor)
            std  = std.clamp(min=1e-3, max=10.0)

            normal = Normal(mean, std)
            u      = normal.rsample()              # grad-friendly sample
            action = torch.tanh(u)

            logp_u = normal.log_prob(u)
            # log π(a) po tanh, patrz SAC Appendix C
            logp   = (logp_u - 2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)

            value  = self.critic(obs_tensor)       # shape ()

        return (
            action.cpu().numpy(),
            logp.item(),
            value.item(),
            u.cpu().numpy()
        )

    # --------------------------------------------------------
    #  Gromadzenie nagród (zachowano bez zmian)
    # --------------------------------------------------------
    def store_reward(self, reward, done):
        self.rew_buf.append(reward)
        self.done_buf.append(done)

    # --------------------------------------------------------
    #  Aktualizacja PPO
    # --------------------------------------------------------
    def learn(self):
        device = self.device  # skrót

        obs   = torch.as_tensor(np.array(self.obs_buf), dtype=torch.float32, device=device)
        acts  = torch.as_tensor(np.array(self.act_buf), dtype=torch.float32, device=device)
        us    = torch.as_tensor(np.array(self.u_buf),   dtype=torch.float32, device=device)
        logp_old = torch.as_tensor(np.array(self.logp_buf), dtype=torch.float32, device=device)

        rewards = torch.as_tensor(self.rew_buf, dtype=torch.float32, device=device)
        dones   = torch.as_tensor(self.done_buf, dtype=torch.float32, device=device)
        values  = torch.as_tensor(self.val_buf, dtype=torch.float32, device=device)

        returns, advantages = self.compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = self.cfg.mini_batch_size
        n          = obs.size(0)
        indices    = torch.randperm(n, device=device)

        for _ in range(self.cfg.ppo_epochs):
            for start in range(0, n, batch_size):
                end = start + batch_size
                idx = indices[start:end]

                b_obs   = obs[idx]
                b_u     = us[idx]
                b_logp_old = logp_old[idx]
                b_returns  = returns[idx]
                b_adv      = advantages[idx]

                mean, std = self.actor(b_obs)
                std = std.clamp(min=1e-3, max=10.0)

                normal      = Normal(mean, std)
                logp_u_new  = normal.log_prob(b_u)
                logp_new    = (logp_u_new - 2 * (np.log(2) - b_u - F.softplus(-2 * b_u))).sum(-1)
                entropy     = normal.entropy().sum(-1)

                ratio  = torch.exp(logp_new - b_logp_old)      # π_new / π_old
                surr1  = ratio * b_adv
                surr2  = torch.clamp(ratio,
                                     1.0 - self.cfg.eps_clip,
                                     1.0 + self.cfg.eps_clip) * b_adv
                actor_loss  = -torch.min(surr1, surr2).mean()

                v_pred = self.critic(b_obs)
                critic_loss = F.mse_loss(v_pred, b_returns)

                loss = (actor_loss
                        + self.cfg.value_loss_coef * critic_loss
                        - self.cfg.entropy_coef   * entropy.mean())

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    max_norm=0.5
                )
                self.optimizer.step()

        wandb.log({"std_mean": std.mean().item()}, step=self.episode)
        return critic_loss.item(), -actor_loss.item()

    # --------------------------------------------------------
    #  GAE + obliczenia pomocnicze
    # --------------------------------------------------------
    def compute_gae(self, rewards, values, dones):
        gae     = 0.0
        gamma   = self.cfg.gamma
        lam     = self.cfg.gae_lambda

        values = torch.cat([values, torch.zeros(1, device=self.device)])  # V_{T}=0
        returns = torch.zeros_like(rewards, device=self.device)

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae   = delta + gamma * lam * (1 - dones[t]) * gae
            returns[t] = gae + values[t]

        advantages = returns - values[:-1]
        return returns, advantages

    # --------------------------------------------------------
    #  Czyszczenie buforów
    # --------------------------------------------------------
    def reset_buffer(self):
        self.obs_buf.clear()
        self.act_buf.clear()
        self.u_buf.clear()
        self.logp_buf.clear()
        self.rew_buf.clear()
        self.done_buf.clear()
        self.val_buf.clear()
