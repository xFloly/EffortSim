import torch
import numpy as np
from pettingzoo.sisl import multiwalker_v9
from utils.common import set_seed
from utils.load_model import load_agent
from utils.load_model import load_checkpoints
from agents.maddpg import MADDPG
from utils.maddpg_io import load_maddpg_checkpoints

def evaluate(cfg, num_episodes=10, max_cycles=500):

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}, seed: {cfg.seed}")
    env = multiwalker_v9.parallel_env()

    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]
    print(f"[eval] Environment loaded: {len(agent_ids)} agents")

    if cfg.agent.lower() == "maddpg":
        maddpg = MADDPG(agent_ids, obs_dim, action_dim, device, cfg)
        if cfg.checkpoint.enabled and cfg.checkpoint.resume:
            load_maddpg_checkpoints(maddpg, agent_ids, cfg)
        agent_type = "maddpg"
    else:
        agents = load_agent(cfg.agent, cfg.checkpoint.path, env, device = device, cfg = cfg)
        agent_type = cfg.agent.lower()

    episode_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0

        while step_count < max_cycles and env.agents:
            step_count += 1

            if agent_type == "maddpg":
                actions = maddpg.act(obs, noise_std=0.0)
            elif agent_type in ["ddpg", "ppo"]:
                actions = {aid: agents.act(obs[aid])[0] for aid in env.agents if aid in obs}
            else:
                raise ValueError(f"[eval] Unsupported agent type: {agent_type}")

            next_obs, rewards, terminations, truncations, infos = env.step(actions)
            dones = {aid: terminations.get(aid, False) or truncations.get(aid, False) for aid in env.agents}

            total_reward += sum(rewards.values())
            obs = next_obs
            if not env.agents:
                break

        episode_rewards.append(total_reward)
        print(f"[eval] Episode {ep+1}/{num_episodes}: Total Reward = {total_reward:.2f}")


    results = {
        "avg_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
    }
    print("[eval] Summary:", results)
    env.close()
    return results
