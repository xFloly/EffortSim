import torch
import numpy as np
from pettingzoo.sisl import multiwalker_v9
from agents.dqn_agent import DQNAgent
from utils.load_model import load_checkpoints

def evaluate(cfg, num_episodes=10, max_cycles=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = multiwalker_v9.parallel_env()
    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]

    agents = {
        aid: DQNAgent(aid, obs_dim, action_dim, device, cfg)
        for aid in agent_ids
    }

    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        load_checkpoints(agents, agent_ids, cfg)

    episode_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0

        for _ in range(max_cycles):
            actions = {
                aid: agents[aid].act(obs[aid])
                for aid in env.agents if aid in obs
            }

            obs, rewards, dones, truncs, infos = env.step(actions)
            total_reward += sum(rewards.values())

            if not env.agents:
                break

        episode_rewards.append(total_reward)

    avg_reward = np.mean(episode_rewards)
    print(f"[eval] Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    env.close()
    return episode_rewards
