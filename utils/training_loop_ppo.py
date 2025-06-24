import torch
import wandb
import numpy as np
from tqdm import trange
from pettingzoo.sisl import multiwalker_v9
from omegaconf import OmegaConf
import os
import random

from agents.ppo import PPOAgent
from utils.metrics import compute_distance, update_agent_effort, log_metrics_to_wandb, penalty
from utils.load_model import load_checkpoints
from utils.common import set_seed

def run(cfg):
    ### WandB initialization ###
    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=cfg.get("wandb_id", None),
        name=cfg.get("wandb_id", None),
        resume="allow"
    )
    print(f"[wandb] Logging to project: {cfg.project} (entity: {cfg.entity})")

    ### Set SEED for reproducibility ###
    set_seed(cfg.seed)
    print(f"[seed] Random seed set to: {cfg.seed}")

    ### Environment Setup ###
    env = multiwalker_v9.parallel_env()
    action_space = env.action_space(env.possible_agents[0])
    print(f"Action space: {action_space}")
    print(f"Low: {action_space.low}")
    print(f"High: {action_space.high}")

    obs, _ = env.reset()
    print(f"[env] Loaded: multiwalker_v9 with {len(env.possible_agents)} agents")

    ### Setup DEVICE ###
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")

    ### Agent and Environment Setup ###
    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]
    print(f"[obs] obs_dim: {obs_dim}, action_dim: {action_dim}")

    ### Shared PPO Agent ###
    shared_agent = PPOAgent("shared", obs_dim, action_dim, device, cfg)
    print(f"[agents] Shared agent initialized")

    ### Resume Checkpoint ###
    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        print("[checkpoint] Attempting to load checkpoint...")
        if load_checkpoints({"shared": shared_agent}, ["shared"], cfg):
            total_steps = shared_agent.steps_done
            start_episode = shared_agent.episode
            print(f"[resume] Loaded checkpoint, starting from episode {start_episode},, total_steps: {total_steps}")
        else:
            total_steps = 0
            start_episode = 0
            print(f"[resume] No checkpoint found, starting fresh")
    else:
        total_steps = 0
        start_episode = 0

    ### Training Loop ###
    reward_history = []

    for episode in trange(start_episode, cfg.num_episodes, desc="Training",initial=start_episode, total=cfg.num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        steps_in_episode = 0

        agent_efforts = {aid: 0.0 for aid in agent_ids}
        agent_rewards = {aid: 0.0 for aid in agent_ids}
        prev_positions = {aid: (0.0, 0.0) for aid in agent_ids}


        done = False
        while not done and steps_in_episode < cfg.max_steps:
            total_steps += 1
            steps_in_episode += 1

            # Select actions for each active agent 
            actions = {
                aid: shared_agent.act(obs[aid])
                for aid in env.agents if aid in obs
            }

            # Environment step
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # Prepare termination flags (for paraller env)
            dones = {
                aid: terminations.get(aid, False) or truncations.get(aid, False)
                for aid in agent_ids
            }

            # Update agent positions for effort tracking 
            current_positions = {
                aid: infos.get(aid, {}).get("walker_pos", prev_positions[aid])
                for aid in env.agents
            }

            # Training step for each active agen
            for aid in actions.keys():
                if aid not in obs or aid not in next_obs:
                    continue

                prev = prev_positions.get(aid, (0.0, 0.0))
                curr = current_positions.get(aid, prev) 
                dist = compute_distance(prev, curr)
                update_agent_effort(agent_efforts, aid, dist)
                prev_positions[aid] = curr

                agent_rewards[aid] += rewards.get(aid, 0.0) + penalty(curr,prev)
                shared_agent.store_reward(rewards[aid], dones[aid])

                # if "done_reason" in infos.get(aid, {}):
                #     print(f"[DEBUG DONE] {aid} done_reason: {infos[aid]['done_reason']}")
                # print(f"[DEBUG REWARD] {aid} reward: {rewards.get(aid, 0.0)}")

            obs = next_obs
            episode_reward += sum(rewards.values())

            # Break if all agents terminated 
            if not env.agents:
                break

        reward_history.append(episode_reward)

        losses = shared_agent.learn()
        if losses is not None:
            episode_loss_critic, episode_loss_actor = losses
        else:
            episode_loss_critic = episode_loss_actor = 0.0

        avg_effort = np.mean(list(agent_efforts.values()))

        ### LOG to wandb ###
        wandb.log({
            "episode": episode,
            "total_reward": episode_reward,
            "avg_critic_loss": episode_loss_critic,
            "avg_actor_loss": episode_loss_actor,
            "avg_effort": avg_effort,
            "steps_total": total_steps
        }, step=episode)

        ### after log_freq episodes : Print LOGS ###
        if (episode + 1) % cfg.log_freq == 0:
            avg_reward = np.mean(reward_history[-cfg.log_freq:])
            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Critic Loss: {episode_loss_critic:.4f} | Actor Loss: {episode_loss_actor:.4f}")
            log_metrics_to_wandb(agent_efforts, agent_rewards, episode)

        ### after checkpoint_freq episodes : Add Checkpoint ###
        if cfg.checkpoint.enabled and (episode + 1) % cfg.checkpoint.freq == 0:
            os.makedirs(cfg.checkpoint.path, exist_ok=True)
            torch.save({
                'model_state_dict': shared_agent.model.state_dict(),
                'optimizer_state_dict': shared_agent.optimizer.state_dict(),
                'episode': episode + 1
            }, os.path.join(cfg.checkpoint.path, f"shared_checkpoint_ep{episode+1}.pt"))
            print(f"Checkpoint saved at episode {episode+1}")

    ### Saving model after training###
    if cfg.checkpoint.enabled:
        os.makedirs(cfg.checkpoint.path, exist_ok=True)
        torch.save({
            'model_state_dict': shared_agent.model.state_dict(),
            'optimizer_state_dict': shared_agent.optimizer.state_dict(),
            'episode': cfg.num_episodes
        }, os.path.join(cfg.checkpoint.path, f"shared_checkpoint_final.pt"))
        print(f"[Final] Checkpoint saved")

    env.close()
    wandb.finish()
