import time
import torch
import numpy as np
import random
from pettingzoo.sisl import multiwalker_v9
from agents.dqn_agent import DQNAgent
from utils.load_model import load_checkpoints

def showcase(cfg, num_episodes=3, max_cycles=500):
    ### Set seed for reproducibility ###
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[seed] Random seed set to: {seed}")
    
    ### Set device ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Create parallel PettingZoo environment (with rendering enabled 'human') ###
    env = multiwalker_v9.parallel_env(render_mode="human")
    env.reset()

    ### Setup agent metadata ###
    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]

    ### Initialize agents ###
    agents = {
        aid: DQNAgent(aid, obs_dim, action_dim, device, cfg)
        for aid in agent_ids
    }

    ### Load model checkpoints if resume enabled ###
    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        load_checkpoints(agents, agent_ids, cfg)

    ### Showcase loop: play episodes visually without training ###
    for i_episode in range(num_episodes):
        print(f"\nShowcase Episode {i_episode + 1}")
        obs, _ = env.reset()
        total_reward_episode = 0

        for step in range(max_cycles):
            env.render()
            time.sleep(0.05)

            #Compute greedy actions for all current agents
            actions = {}
            for aid in env.agents:
                if aid in obs:
                    with torch.no_grad():
                        state = np.array(obs[aid], dtype=np.float32)
                        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                        action = agents[aid].policy_net(state_tensor).cpu().numpy().squeeze()
                        action = np.clip(action, -1.0, 1.0).astype(np.float32)
                        actions[aid] = action

            if not actions:
                break

            # Step the environment in parallel
            obs, rewards, dones, truncs, infos = env.step(actions)
            total_reward_episode += sum(rewards.values())

            # End episode if all agents done 
            if all(dones.values()):
                break

        print(f"End of Showcase Episode {i_episode + 1}. Total Reward: {total_reward_episode:.2f}")

    env.close()
