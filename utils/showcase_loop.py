import time
import torch
import numpy as np
from pettingzoo.sisl import multiwalker_v9
from agents.dqn_agent import DQNAgent
from utils.load_model import load_checkpoints

def showcase(cfg, num_episodes=3, max_cycles=500):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = multiwalker_v9.parallel_env(render_mode="human")
    env.reset()

    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]

    agents = {
        aid: DQNAgent(aid, obs_dim, action_dim, device, cfg)
        for aid in agent_ids
    }

    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        load_checkpoints(agents, agent_ids, cfg)

    for i_episode in range(num_episodes):
        print(f"\nShowcase Episode {i_episode + 1}")
        obs, _ = env.reset()
        total_reward_episode = 0

        for step in range(max_cycles):
            env.render()
            time.sleep(0.05)

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

            obs, rewards, dones, truncs, infos = env.step(actions)
            total_reward_episode += sum(rewards.values())

            if all(dones.values()):
                break

        print(f"End of Showcase Episode {i_episode + 1}. Total Reward: {total_reward_episode:.2f}")

    env.close()
