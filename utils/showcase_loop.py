import time
import torch
import numpy as np
import random
from pettingzoo.sisl import multiwalker_v9
import imageio 

from agents.ddpg import DDPGAgent
from utils.load_model import load_checkpoints
from utils.common import set_seed

def showcase(cfg, max_cycles=500):
    ### Set seed for reproducibility ###
    set_seed(cfg.seed)
    print(f"[seed] Random seed set to: {cfg.seed}")
    
    ### Set device ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ### Create parallel PettingZoo environment (with rendering enabled 'human') ###
    env = multiwalker_v9.parallel_env(render_mode="human",
                                      terminate_reward=-100.0,
                                      fall_reward=-10.0,
                                      forward_reward=20.0)
    env.reset()

    ### Setup agent metadata ###
    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]

    ### Initialize agents ###
    agents = {
        aid: DDPGAgent(aid, obs_dim, action_dim, device, cfg)
        for aid in agent_ids
    }

    ### Load model checkpoints if resume enabled ###
    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        load_checkpoints(agents, agent_ids, cfg)

    print("\n[showcase] Running showcase loop continuously.")
    episode = 0

    try:
        while True:
            episode += 1
            print(f"\nShowcase Episode {episode}")
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
                            action = agents[aid].act(state)
                            action = np.clip(action, -1.0, 1.0).astype(np.float32)
                            actions[aid] = action

                if not actions:
                    break

                obs, rewards, dones, truncs, infos = env.step(actions)
                total_reward_episode += sum(rewards.values())

                if all(dones.values()):
                    break

            print(f"End of Showcase Episode {episode}. Total Reward: {total_reward_episode:.2f}")
    except KeyboardInterrupt:
        print("\n[showcase] Showcase interrupted by user.")

    env.close()
