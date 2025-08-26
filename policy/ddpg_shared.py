import torch
import wandb
import numpy as np
from tqdm import trange
from pettingzoo.sisl import multiwalker_v9
from omegaconf import OmegaConf
import os

from agents.ddpg import DDPGAgent
from utils.metrics import compute_distance, update_agent_effort, log_metrics_to_wandb
from utils.load_model import load_checkpoints
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
        forward_reward=20.0)
    obs, _ = env.reset()
    print(f"[env] Loaded: multiwalker_v9 with {len(env.possible_agents)} agents")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")

    ### Agent and Environment Setup ###
    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]
    print(f"[obs] obs_dim: {obs_dim}, action_dim: {action_dim}")

    ### Shared DDPG Agent ###
    shared_agent = DDPGAgent("shared", obs_dim, action_dim, device, cfg)
    print(f"[agents] Shared agent initialized")

    ### Resume Checkpoint ###
    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        load_checkpoints({"shared": shared_agent}, ["shared"], cfg)
        total_steps = shared_agent.steps_done
        start_episode = shared_agent.episode
        print(f"[resume] Resuming from episode {start_episode}, total_steps: {total_steps}")
    else:
        total_steps = 0
        start_episode = 0

    ### Training Loop ###
    reward_history = []

    for episode in trange(start_episode, cfg.num_episodes, desc="Training",initial=start_episode, total=cfg.num_episodes):
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
            shared_agent.steps_done = total_steps

            actions = {
                aid: shared_agent.act(obs[aid])
                for aid in env.agents if aid in obs
            }

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            dones = {
                aid: terminations.get(aid, False) or truncations.get(aid, False)
                for aid in agent_ids
            }

            current_positions = {
                aid: infos.get(aid, {}).get("walker_pos", prev_positions[aid])
                for aid in env.agents
            }

            step_loss_critic = 0.0
            step_loss_actor = 0.0
            active_agents = 0

            for aid in actions.keys():
                if aid not in obs or aid not in next_obs:
                    continue

                state = obs[aid]
                action = actions[aid]
                reward = rewards.get(aid, 0.0)
                next_state = next_obs[aid]
                done = dones.get(aid, False)

                agent_rewards[aid] += reward

                prev = prev_positions.get(aid, (0.0, 0.0))
                curr = current_positions.get(aid, prev) 
                dist = compute_distance(prev, curr)
                update_agent_effort(agent_efforts, aid, dist)
                prev_positions[aid] = curr

                shared_agent.push(state, action, reward, next_state, done)
                losses = shared_agent.learn()
                if losses:
                    step_loss_critic += losses[0]
                    step_loss_actor += losses[1]
                    active_agents += 1

            obs = next_obs
            episode_reward += sum(rewards.values())

            if active_agents > 0:
                episode_loss_critic += step_loss_critic / active_agents
                episode_loss_actor += step_loss_actor / active_agents

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
            os.makedirs(cfg.checkpoint.path, exist_ok=True)
            torch.save({
                'actor_state_dict': shared_agent.actor.state_dict(),
                'actor_target_state_dict': shared_agent.actor_target.state_dict(),
                'critic_state_dict': shared_agent.critic.state_dict(),
                'critic_target_state_dict': shared_agent.critic_target.state_dict(),
                'actor_optimizer_state_dict': shared_agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': shared_agent.critic_optimizer.state_dict(),
                'steps_done': shared_agent.steps_done,
                'episode': episode + 1
            }, os.path.join(cfg.checkpoint.path, f"shared_checkpoint_ep{episode+1}.pt"))
            print(f"[Checkpoint] Shared model saved at episode {episode+1}")

    ### Final Save ###
    if cfg.checkpoint.enabled:
        os.makedirs(cfg.checkpoint.path, exist_ok=True)
        torch.save({
            'actor_state_dict': shared_agent.actor.state_dict(),
            'actor_target_state_dict': shared_agent.actor_target.state_dict(),
            'critic_state_dict': shared_agent.critic.state_dict(),
            'critic_target_state_dict': shared_agent.critic_target.state_dict(),
            'actor_optimizer_state_dict': shared_agent.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': shared_agent.critic_optimizer.state_dict(),
            'steps_done': shared_agent.steps_done,
            'episode': episode + 1
        }, os.path.join(cfg.checkpoint.path, f"shared_checkpoint_final.pt"))
        print(f"[Final] Shared model checkpoint saved.")

    env.close()
    wandb.finish()
