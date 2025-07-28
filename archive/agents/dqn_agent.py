import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.model(x)


class DQNAgent:
    def __init__(self, agent_id, obs_dim, action_dim, device, cfg=None):
        self.agent_id = agent_id
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.policy_net = MLP(obs_dim, action_dim).to(device)
        self.target_net = MLP(obs_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Hyperparameters from config or defaults
        self.lr = cfg.lr if cfg else 1e-3
        self.gamma = cfg.gamma if cfg else 0.99
        self.eps_start = cfg.eps_start if cfg else 1.0
        self.eps_end = cfg.eps_end if cfg else 0.01
        self.eps_decay = cfg.eps_decay if cfg else 0.995
        self.buffer_size = cfg.buffer_size if cfg else 50000
        self.batch_size = cfg.batch_size if cfg else 64

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

        self.replay_buffer = deque(maxlen=self.buffer_size)
        self.epsilon = self.eps_start
        self.steps_done = 0

    def act(self, obs):
        if random.random() < self.epsilon:
            return np.random.uniform(-1.0, 1.0, size=self.action_dim).astype(np.float32)

        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.policy_net(obs_tensor).cpu().numpy().squeeze()

        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def push(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def learn(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.replay_buffer) < batch_size:
            return None

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Efficient conversion of lists of np arrays
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Q(s, a): approximate Q-values based on current policy
        q_values = self.policy_net(states)
        current_q = torch.sum(q_values * actions, dim=1, keepdim=True)  # inner product as approximation

        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.max(dim=1, keepdim=True).values
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)

        return loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
