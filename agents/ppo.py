import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU()
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # zakładając przestrzeń akcji ciągłą [-1, 1]
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared_out = self.shared(x)
        return self.actor(shared_out), self.critic(shared_out)

class PPOAgent:
    def __init__(self, agent_id, obs_dim, action_dim, device, cfg):
        self.agent_id = agent_id
        self.device = device
        self.cfg = cfg

        self.model = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)

        self.obs_buf = []
        self.act_buf = []
        self.logp_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.val_buf = []

    def act(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_mean, value = self.model(obs)
        dist = torch.distributions.Normal(action_mean, torch.ones_like(action_mean) * 0.1)
        # action = dist.sample()
        action = torch.tanh(dist.sample())  # ograniczenie do (-1, 1)

        log_prob = dist.log_prob(action).sum()

        # ZBIERANIE DANYCH DO BUFORÓW
        self.obs_buf.append(obs.cpu().numpy())              # numpy array
        self.act_buf.append(action.cpu().numpy())           # numpy array
        self.logp_buf.append(log_prob.item())               # float (scalar)
        self.val_buf.append(value.item())                   # float (scalar)

        return action.cpu().numpy()


    def store_reward(self, reward, done):
        self.rew_buf.append(reward)
        self.done_buf.append(done)

    def learn(self):
        # Konwertuj dane na tensory
        obs = torch.tensor(np.array(self.obs_buf), dtype=torch.float32).to(self.device)

        actions = torch.tensor(np.array(self.act_buf), dtype=torch.float32).to(self.device)
        log_probs_old = torch.tensor(np.array(self.logp_buf), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(self.rew_buf, dtype=torch.float32).to(self.device)
        dones = torch.tensor(self.done_buf, dtype=torch.float32).to(self.device)
        values = torch.tensor(self.val_buf, dtype=torch.float32).to(self.device)


        returns, advantages = self.compute_gae(rewards, values, dones)

        for _ in range(self.cfg.ppo_epochs):
            action_mean, value = self.model(obs)
            dist = torch.distributions.Normal(action_mean, torch.ones_like(action_mean) * 0.1)
            log_probs = dist.log_prob(actions).sum(axis=1)
            entropy = dist.entropy().sum(axis=1)

            ratio = torch.exp(log_probs - log_probs_old)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.cfg.eps_clip, 1.0 + self.cfg.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = nn.MSELoss()(value.squeeze(), returns)

            loss = actor_loss + self.cfg.value_loss_coef * critic_loss - self.cfg.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.reset_buffer()
        return critic_loss.item(), -actor_loss.item()

    def compute_gae(self, rewards, values, dones):
        gae = 0
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda
        values = values.tolist() + [0]
        returns = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            returns.insert(0, gae + values[t])
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        advantages = returns - torch.tensor(values[:-1], dtype=torch.float32).to(self.device)
        return returns, advantages

    def reset_buffer(self):
        self.obs_buf.clear()
        self.act_buf.clear()
        self.logp_buf.clear()
        self.rew_buf.clear()
        self.done_buf.clear()
        self.val_buf.clear()
