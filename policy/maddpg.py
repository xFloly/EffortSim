import torch
import wandb
import numpy as np
from tqdm import trange
from pettingzoo.sisl import multiwalker_v9
from omegaconf import OmegaConf
import os

from agents.maddpg import MADDPG 
from utils.metrics import compute_distance, update_agent_effort, log_metrics_to_wandb, penalty
from utils.maddpg_io import save_maddpg_checkpoints, load_maddpg_checkpoints
from utils.common import set_seed


def run(cfg):
    ### WandB initialization ###
    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=cfg.wandb_id,
        name=cfg.wandb_id,
        resume='allow'
    )
    print(f"[wandb] Logging to project: {cfg.project} (entity: {cfg.entity})")
    
    ### Set SEED for reproducibility ###
    set_seed(cfg.seed)
    print(f"[seed] Random seed set to: {cfg.seed}")

    ### Environment Setup ###
    env = multiwalker_v9.parallel_env(
        terminate_reward=-100.0,
        fall_reward=-10.0,
        forward_reward=20.0
    )
    obs, _ = env.reset()
    agent_ids = env.possible_agents
    print(f"[env] Loaded: multiwalker_v9 with {len(agent_ids)} agents")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")

    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]
    print(f"[obs] obs_dim: {obs_dim}, action_dim: {action_dim}")

    ### MADDPG Agent ###
    maddpg = MADDPG(agent_ids, obs_dim, action_dim, device, cfg)
    print(f"[agents] MADDPG initialized with centralized critic and {len(agent_ids)} actors")

    #  Resume from checkpoint 
    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        loaded_any, total_steps, start_episode = load_maddpg_checkpoints(maddpg, agent_ids, cfg)
        if loaded_any:
            print(f"[resume] Resuming from episode {start_episode}, total_steps: {total_steps}")
        else:
            total_steps, start_episode = 0, 0
    else:
        total_steps, start_episode = 0, 0

    ### Training Loop ###
    reward_history = []

    for episode in trange(start_episode, cfg.num_episodes, desc="Training"):
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
            total_steps += 1
            steps_in_episode += 1

            #Actions from MADDPG 
            actions = maddpg.act(obs)

            #Step environment 
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            dones = {aid: terminations.get(aid, False) or truncations.get(aid, False) for aid in agent_ids}

            #Effort & reward tracking 
            current_positions = {aid: infos.get(aid, {}).get("walker_pos", prev_positions[aid]) for aid in env.agents}
            for aid in actions.keys():
                
                base_reward = rewards.get(aid, 0.0)
                prev = prev_positions.get(aid, (0.0, 0.0))
                curr = current_positions.get(aid, prev)
                r_penalty = penalty(curr, prev)
                total_reward = base_reward + r_penalty

                agent_rewards[aid] += total_reward

                update_agent_effort(agent_efforts, aid, compute_distance(prev, curr))
                prev_positions[aid] = curr

            #Store transition in shared replay
            maddpg.push(obs, actions, rewards, next_obs, dones)

            #Learn (central critic + all actors)
            losses = maddpg.learn()
            if losses:
                critic_loss, actor_loss = losses
                episode_loss_critic += critic_loss
                episode_loss_actor += actor_loss

            obs = next_obs
            episode_reward += sum(rewards.values())

            if not env.agents:
                break

        reward_history.append(episode_reward)

        avg_critic_loss = episode_loss_critic / steps_in_episode if steps_in_episode else 0
        avg_actor_loss = episode_loss_actor / steps_in_episode if steps_in_episode else 0

        wandb.log({
            "episode": episode,
            "total_reward": episode_reward,
            "avg_critic_loss": avg_critic_loss,
            "avg_actor_loss": avg_actor_loss,
            "steps_total": total_steps
        }, step=episode)

        if (episode + 1) % cfg.log_freq == 0:
            avg_reward = np.mean(reward_history[-cfg.log_freq:])
            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Critic Loss: {avg_critic_loss:.4f} | Actor Loss: {avg_actor_loss:.4f}")
            log_metrics_to_wandb(agent_efforts, agent_rewards, episode)

        if cfg.checkpoint.enabled and (episode + 1) % cfg.checkpoint_freq == 0:
            save_maddpg_checkpoints(maddpg, agent_ids, cfg, episode, total_steps)

      

    # final save 
    save_maddpg_checkpoints(maddpg, agent_ids, cfg, episode, total_steps, final=True)

    env.close()
    wandb.finish()
