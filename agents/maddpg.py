# agents/maddpg.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from agents.ddpg import Actor, Critic

class MADDPG:
    def __init__(self, agent_ids, obs_dim, action_dim, device, cfg=None):
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # hyperparams
        self.lr_actor = cfg.lr if cfg else 1e-3
        self.lr_critic = cfg.lr if cfg else 1e-3
        self.gamma = cfg.gamma if cfg else 0.95
        self.tau = 0.01
        self.batch_size = cfg.batch_size if cfg else 64
        self.buffer_size = cfg.buffer_size if cfg else 1000000

        # one actor per agent
        self.actors = {aid: Actor(obs_dim, action_dim).to(device) for aid in agent_ids}
        self.actors_target = {aid: Actor(obs_dim, action_dim).to(device) for aid in agent_ids}
        for aid in agent_ids:
            self.actors_target[aid].load_state_dict(self.actors[aid].state_dict())

        self.actor_optimizers = {aid: optim.Adam(self.actors[aid].parameters(), lr=self.lr_actor) 
                                 for aid in agent_ids}

        # centralized critic
        self.critic = Critic(self.n_agents * obs_dim, self.n_agents * action_dim).to(device)
        self.critic_target = Critic(self.n_agents * obs_dim, self.n_agents * action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # replay buffer
        self.replay = deque(maxlen=self.buffer_size)

    def act(self, obs_dict, noise_std=0.2):
        """Each agent acts with its own actor. obs_dict is {aid: obs}"""
        actions = {}
        for aid, obs in obs_dict.items():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                act = self.actors[aid](obs_t).cpu().numpy().squeeze()
            noise = np.random.normal(0, noise_std, size=self.action_dim)
            act = np.clip(act + noise, -1, 1)
            actions[aid] = act.astype(np.float32)
        return actions

    def push(self, obs, actions, rewards, next_obs, dones):
        """Store transition in replay"""
        # obs/actions are dicts
        g_state = np.concatenate([obs[aid] for aid in self.agent_ids], axis=-1)
        g_action = np.concatenate([actions[aid] for aid in self.agent_ids], axis=-1)
        g_next_state = np.concatenate([next_obs.get(aid, np.zeros_like(obs[aid])) for aid in self.agent_ids], axis=-1)
        r_vec = np.array([rewards.get(aid, 0.0) for aid in self.agent_ids], dtype=np.float32)
        d_vec = np.array([dones.get(aid, 0.0) for aid in self.agent_ids], dtype=np.float32)
        self.replay.append((g_state, g_action, r_vec, g_next_state, d_vec, obs, next_obs))

    def learn(self):
        if len(self.replay) < self.batch_size:
            return None

        batch = random.sample(self.replay, self.batch_size)
        g_s, g_a, r_vec, g_ns, d_vec, obs_batch, next_obs_batch = zip(*batch)

        g_s = torch.tensor(np.array(g_s), dtype=torch.float32, device=self.device)
        g_a = torch.tensor(np.array(g_a), dtype=torch.float32, device=self.device)
        r_vec = torch.tensor(np.array(r_vec), dtype=torch.float32, device=self.device)  # (B,n_agents)
        g_ns = torch.tensor(np.array(g_ns), dtype=torch.float32, device=self.device)
        d_vec = torch.tensor(np.array(d_vec), dtype=torch.float32, device=self.device)

        B = g_s.size(0)

        # ---- critic update ----
        with torch.no_grad():
            next_actions = []
            ns = torch.reshape(g_ns, (B, self.n_agents, self.obs_dim))
            for i, aid in enumerate(self.agent_ids):
                a_next = self.actors_target[aid](ns[:, i, :])
                next_actions.append(a_next)
            next_joint_a = torch.cat(next_actions, dim=1)
            q_next = self.critic_target(g_ns, next_joint_a)
            team_r = r_vec.mean(dim=1, keepdim=True)
            team_done = d_vec.max(dim=1, keepdim=True).values
            y = team_r + (1 - team_done) * self.gamma * q_next

        q = self.critic(g_s, g_a)
        critic_loss = nn.MSELoss()(q, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- actor update (per-agent) ----
        actor_losses = []
        s_split = torch.reshape(g_s, (B, self.n_agents, self.obs_dim)).detach()
        for i, aid in enumerate(self.agent_ids):
            # replace only this agent’s action
            a_list = []
            for j, aj in enumerate(self.agent_ids):
                if j == i:
                    a_i = self.actors[aid](s_split[:, i, :])
                    a_list.append(a_i)
                else:
                    a_list.append(g_a[:, j*self.action_dim:(j+1)*self.action_dim].detach())
            joint_a_pi = torch.cat(a_list, dim=1)
            loss = -self.critic(g_s, joint_a_pi).mean()
            self.actor_optimizers[aid].zero_grad()
            loss.backward()
            self.actor_optimizers[aid].step()
            actor_losses.append(loss.item())

        # soft update
        self._soft_update(self.critic, self.critic_target)
        for aid in self.agent_ids:
            self._soft_update(self.actors[aid], self.actors_target[aid])

        return critic_loss.item(), np.mean(actor_losses)

    def _soft_update(self, net, target_net):
        for p, tp in zip(net.parameters(), target_net.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
