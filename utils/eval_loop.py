import torch
import numpy as np
import random
from pettingzoo.sisl import multiwalker_v9

from agents.dqn_agent import DQNAgent
from utils.load_model import load_checkpoints
from utils.common import set_seed

def evaluate(cfg, num_episodes=10, max_cycles=500):
    ### Set seed for reproducibility ###
    set_seed(cfg.seed)
    print(f"[seed] Random seed set to: {cfg.seed}")

    ### Set device ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create parallel PettingZoo environment (multiwalker) 
    env = multiwalker_v9.parallel_env()
    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]

    ### Initialize agents ###
    agents = {
        aid: DQNAgent(aid, obs_dim, action_dim, device, cfg)
        for aid in agent_ids
    }

    ### Load pre-trained model checkpoints if resume is enabled ###
    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        load_checkpoints(agents, agent_ids, cfg)

    episode_rewards = []

    ### Evaluation loop (no training, just forward passes) ###
    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0

        for _ in range(max_cycles):
            # Collect actions from all active agents in parallel
            actions = {
                aid: agents[aid].act(obs[aid])
                for aid in env.agents if aid in obs
            }

            # Step the parallel environment
            obs, rewards, dones, truncs, infos = env.step(actions)
            total_reward += sum(rewards.values())

            # break when all agents are done
            if not env.agents:
                break

        episode_rewards.append(total_reward)

    ### Compute and print final evaluation score ###
    results = {
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards))
    }
    print("[eval] Summary:", results)
    return results

    env.close()
    return episode_rewards
