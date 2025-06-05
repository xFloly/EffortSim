import torch
import numpy as np
import random
from tqdm import trange
from pettingzoo.sisl import multiwalker_v9

from agents.ddpg import DDPGAgent
from utils.metrics import compute_distance, update_agent_effort, log_metrics_to_wandb
from utils.load_model import load_checkpoints

def run(cfg):
    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env = multiwalker_v9.parallel_env()
    obs, _ = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]

    agents = {aid: DDPGAgent(aid, obs_dim, action_dim, device, cfg) for aid in agent_ids}

    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        load_checkpoints(agents, agent_ids, cfg)

    reward_history = []
    total_steps = 0

    for episode in trange(cfg.num_episodes, desc="Training"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_loss_critic = 0.0
        episode_loss_actor = 0.0
        steps_in_episode = 0

        agent_efforts = {aid: 0.0 for aid in agent_ids}
        agent_rewards = {aid: 0.0 for aid in agent_ids}
        prev_positions = {aid: (0.0, 0.0) for aid in agent_ids}

        done = False
        while not done and steps_in_episode < cfg.max_steps:
            actions = {}
            for aid in agent_ids:
                actions[aid] = agents[aid].act(obs[aid])

            next_obs, rewards, dones, truncations, infos = env.step(actions)
            done = any(dones.values())

            # Update efforts by tracking agent position changes (assumes pos in obs[:2])
            for aid in agent_ids:
                current_pos = obs[aid][:2]
                update_agent_effort(agent_efforts, aid, compute_distance(prev_positions[aid], current_pos))
                prev_positions[aid] = current_pos
                agent_rewards[aid] += rewards[aid]

            # Store transitions for all agents
            for aid in agent_ids:
                agents[aid].push(obs[aid], actions[aid], rewards[aid], next_obs[aid], dones[aid])

            # Learn for all agents
            for aid in agent_ids:
                losses = agents[aid].learn()
                if losses:
                    episode_loss_critic += losses[0]
                    episode_loss_actor += losses[1]

            obs = next_obs
            episode_reward += sum(rewards.values())
            steps_in_episode += 1
            total_steps += 1

            # Optional: log every few steps, e.g. to wandb or console

        reward_history.append(episode_reward)

        if episode % cfg.log_freq == 0:
            avg_reward = np.mean(reward_history[-cfg.log_freq:])
            avg_critic_loss = episode_loss_critic / steps_in_episode if steps_in_episode else 0
            avg_actor_loss = episode_loss_actor / steps_in_episode if steps_in_episode else 0
            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Critic Loss: {avg_critic_loss:.4f} | Actor Loss: {avg_actor_loss:.4f}")
            log_metrics_to_wandb(agent_efforts, agent_rewards, episode)

        # Save checkpoint
        if cfg.checkpoint.enabled and episode % cfg.checkpoint_freq == 0:
            for aid in agent_ids:
                ckpt_path = f"{cfg.checkpoint.path}/{aid}_checkpoint_ep{episode}.pt"
                torch.save({
                    'actor_state_dict': agents[aid].actor.state_dict(),
                    'actor_target_state_dict': agents[aid].actor_target.state_dict(),
                    'critic_state_dict': agents[aid].critic.state_dict(),
                    'critic_target_state_dict': agents[aid].critic_target.state_dict(),
                    'actor_optimizer_state_dict': agents[aid].actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agents[aid].critic_optimizer.state_dict(),
                    'steps_done': agents[aid].steps_done
                }, ckpt_path)
                print(f"Saved checkpoint for {aid} at episode {episode}")

    env.close()

    # Visualization after training:
    import matplotlib.pyplot as plt
    plt.plot(range(len(reward_history)), reward_history)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Training Reward over Episodes")
    plt.show()
