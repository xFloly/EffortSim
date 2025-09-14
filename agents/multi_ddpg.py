# agents/multi_ddpg.py
import numpy as np
from agents.ddpg import DDPGAgent
from networkx.algorithms.bipartite.cluster import modes
from utils.load_model import load_checkpoints

class MultiDDPG:
    def __init__(self, agent_ids, obs_dim, action_dim, device="cpu", cfg=None, checkpoint_path=None):
        self.device = device
        self.agent_ids = agent_ids
        self.agents = {
            aid: DDPGAgent(aid, obs_dim, action_dim, device=device, cfg=cfg)
            for aid in agent_ids
        }

        loaded = load_checkpoints(self.agents, agent_ids, cfg, checkpoint_path)
        if not loaded:
            print("[warn] No checkpoints loaded for MultiDDPG (agents start untrained)")

    def act(self, obs, noise_std=0.0):
        """Return actions for all agents. Adds optional Gaussian noise."""
        actions = {}
        for aid in obs.keys():
            action = self.agents[aid].act(obs[aid])  # original DDPGAgent act
            if noise_std > 0.0:
                action += np.random.normal(0, noise_std, size=action.shape)
            actions[aid] = action
        return actions

    def eval(self):
        for agent in self.agents.values():
            agent.actor.eval()
            if hasattr(agent, "critic"):
                agent.critic.eval()
