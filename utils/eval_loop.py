import time
import torch
import numpy as np
from pettingzoo.sisl import multiwalker_v9
from utils.common import set_seed
from utils.load_model import load_agent, load_checkpoints, load_maddpg_checkpoints
from agents.maddpg import MADDPG
from agents.ddpg import DDPGAgent
from agents.ppo import PPOAgent



def evaluate(cfg, agent_name, model_path, num_episodes=1, max_cycles=500):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}, seed: {cfg.seed}")

    env = multiwalker_v9.parallel_env(render_mode="rgb_array",
                                      terminate_reward=-100.0,
                                      fall_reward=-10.0,
                                      forward_reward=20.0)
    obs, _ = env.reset()
    agent_name = agent_name.lower()
    multi_agent = False
    agent_ids  = env.possible_agents
    obs_space = env.observation_space(agent_ids[0])
    obs_dim = obs_space.shape[0]  # if Box space
    action_space = env.action_space(agent_ids[0])
    action_dim = action_space.shape[0]

    if agent_name == "maddpg":
        from agents.maddpg import MADDPG
        agent = MADDPG(agent_ids, obs_dim, action_dim, device=device, cfg=cfg)
        multi_agent = True
        load_maddpg_checkpoints(agent, agent_ids, cfg, model_path)
    elif agent_name == "ddpg":
        from agents.multi_ddpg import MultiDDPG
        agent = MultiDDPG(agent_ids, obs_dim, action_dim, device=device, cfg=cfg, checkpoint_path=model_path)
        multi_agent = True
    elif agent_name == "ppo":
        agent = load_agent("ppo", model_path, env, device = device, cfg = cfg)
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")


    # ----- Rollout loop -----
    data = [None for _ in range(num_episodes)]

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        reward_total = {aid: 0.0 for aid in agent_ids}
        dist = {aid: 0.0 for aid in agent_ids}

        for _ in range(max_cycles):
            steps += 1
            if multi_agent:
                actions = agent.act(obs, noise_std=0.0)
            elif agent_name == "ddpg":
                actions = {
                aid: agent.act(obs[aid])
                for aid in env.agents if aid in obs
            }
            else:  # PPO
                actions = {
                    aid: agent.act(obs[aid])[0]
                    for aid in env.agents if aid in obs
                }

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # Calculate distance
            for aid in env.agents:
                if aid in next_obs:
                    vx, vy = float(next_obs[aid][2]), float(next_obs[aid][3])
                    dist[aid] += (vx**2 + vy**2) ** 0.5
                if aid in rewards:
                    reward_total[aid] += rewards[aid]

            obs = next_obs
            done = not env.agents or all(terminations.values()) or all(truncations.values())
            if done:
                break

        data[ep] = (steps, dist,reward_total)

    # ---- Aggregate Results ----
    print("\n[✓] Evaluation complete, Aggregating Results.\n")

    
    single_walker_dist = {aid: 0.0 for aid in agent_ids}
    single_walker_rewards = {aid: 0.0 for aid in agent_ids}
    total_steps = 0
    total_distance = 0.0
    total_reward = 0.0

    for steps, dist, rewards in data:
        total_steps += steps
        for aid in agent_ids:
            single_walker_dist[aid] += dist[aid]
            single_walker_rewards[aid] += rewards[aid]
            total_distance += dist[aid]
            total_reward += rewards[aid]

    avg_steps = total_steps / num_episodes
    avg_dist = total_distance / (num_episodes * len(agent_ids))
    avg_reward = total_reward / (num_episodes * len(agent_ids))

    # Per-agent stats
    for aid in agent_ids:
        print(f"  {aid}: {single_walker_dist[aid] / num_episodes:.3f} units,"
              f" {single_walker_rewards[aid] / num_episodes:.3f} reward")

    print(f"\n[eval] Total distance traveled: {total_distance:.3f} units")
    print(f"[eval] Average distance per walker/episode: {avg_dist:.3f} units")
    print(f"[eval] Total reward: {total_reward:.3f}")
    print(f"[eval] Average reward per walker/episode: {avg_reward:.3f}")
    print(f"[eval] Average steps per episode: {avg_steps:.1f}")
    print(f"[eval] Average speed: {avg_dist / avg_steps:.4f} units/action")

    env.close()
