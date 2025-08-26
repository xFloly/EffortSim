import torch
import wandb
import numpy as np
from tqdm import trange
from pettingzoo.sisl import multiwalker_v9
from omegaconf import OmegaConf
import os
import random

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

    ### Environment Setup for multiwalker with stick  in PARALLEL###
    env = multiwalker_v9.parallel_env()
    obs, _ = env.reset()
    print(f"[env] Loaded: multiwalker_v9 with {len(env.possible_agents)} agents")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")

    ### Initialize Agents ### 
    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]
    print(f"[obs] obs_dim: {obs_dim}, action_dim: {action_dim}")

    ### Choose Agent ###
    agents = {
        aid: DDPGAgent(aid, obs_dim, action_dim, device, cfg)
        for aid in agent_ids
    }
    print(f"[agents] Initialized: {', '.join(agent_ids)}")

    ### Load pre-trained model checkpoints if resume is enabled ###
    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        load_checkpoints(agents, agent_ids, cfg)
        # Estimate the corresponding episode to resume from
        total_steps = max(agent.steps_done for agent in agents.values())
        start_episode = max(agent.episode for agent in agents.values())
        print(f"[resume] Resuming from episode {start_episode}, total_steps: {total_steps}")
    else:
        total_steps = 0
        start_episode = 0

    ### Train Loop
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

            # Update global steps for epsilon decay
            for agent in agents.values():
                agent.steps_done = total_steps

            # Select actions for each active agent 
            actions = {
                aid: agents[aid].act(obs[aid])
                for aid in env.agents if aid in obs
            }

            # Environment step
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # Prepare termination flags (for paraller env)
            dones = {
                aid: terminations.get(aid, False) or truncations.get(aid, False)
                for aid in agents.keys()
            }

            # Update agent positions for effort tracking 
            current_positions = {
                aid: infos.get(aid, {}).get("walker_pos", prev_positions[aid])
                for aid in env.agents
            }

            # Training step for each active agen
            step_loss_critic = 0
            step_loss_actor = 0
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

                # Skpis dead/fallen agents
                if aid in current_positions: 
                    prev = prev_positions.get(aid, (0.0, 0.0))
                    curr = current_positions[aid]
                    dist = compute_distance(prev, curr)
                    update_agent_effort(agent_efforts, aid, dist)
                    prev_positions[aid] = curr

                agents[aid].push(state, action, reward, next_state, done)
                losses = agents[aid].learn()
                if losses is not None:
                    step_loss_critic += losses[0]
                    step_loss_actor += losses[1]
                    active_agents += 1

            obs = next_obs
            episode_reward += sum(rewards.values())
            if active_agents > 0:
                episode_loss_critic += step_loss_critic / active_agents
                episode_loss_actor += step_loss_actor / active_agents


            # Break if all agents terminated 
            if not env.agents:
                break

        reward_history.append(episode_reward)
 
        # #after `target_update_freq` episodes update the agents:
        # if (episode + 1) % cfg.target_update_freq == 0:
        #     for agent in agents.values():
        #         agent.update_target_net()


        avg_critic_loss = episode_loss_critic / steps_in_episode if steps_in_episode else 0
        avg_actor_loss = episode_loss_actor / steps_in_episode if steps_in_episode else 0

        ### LOG to wandb ###
        wandb.log({
            "episode": episode,
            "total_reward": episode_reward,
            "avg_critic_loss": avg_critic_loss,
            "avg_actor_loss": avg_actor_loss,
            # "epsilon": agents[agent_ids[0]].epsilon,
            "steps_total": total_steps
        }, step=episode)

        ### after log_freq episodes : Print LOGS ###
        if (episode + 1) % cfg.log_freq == 0:
            avg_reward = np.mean(reward_history[-cfg.log_freq:])
            avg_critic_loss = episode_loss_critic / steps_in_episode if steps_in_episode else 0
            avg_actor_loss = episode_loss_actor / steps_in_episode if steps_in_episode else 0
            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Critic Loss: {avg_critic_loss:.4f} | Actor Loss: {avg_actor_loss:.4f}")
            log_metrics_to_wandb(agent_efforts, agent_rewards, episode)

        ### after checkpoint_freq episodes : Add Checkpoint ###
        if cfg.checkpoint.enabled and (episode + 1) % cfg.checkpoint_freq == 0:
            os.makedirs(cfg.checkpoint.path, exist_ok=True)
            for aid in agent_ids:
                ckpt = {
                    'actor_state_dict': agents[aid].actor.state_dict(),
                    'actor_target_state_dict': agents[aid].actor_target.state_dict(),
                    'critic_state_dict': agents[aid].critic.state_dict(),
                    'critic_target_state_dict': agents[aid].critic_target.state_dict(),
                    'actor_optimizer_state_dict': agents[aid].actor_optimizer.state_dict(),
                    'critic_optimizer_state_dict': agents[aid].critic_optimizer.state_dict(),
                    # 'epsilon': agents[aid].epsilon,
                    'steps_done': agents[aid].steps_done,
                    'episode': episode + 1
                }
                torch.save(ckpt, os.path.join(cfg.checkpoint.path, f"{aid}_checkpoint_ep{episode+1}.pt"))
                print(f"Checkpoint saved for {aid} episode {episode+1} ")


    ### Saving model after training###
    if cfg.checkpoint.enabled:
        os.makedirs(cfg.checkpoint.path, exist_ok=True)
        for aid in agent_ids:
            ckpt = {
                'actor_state_dict': agents[aid].actor.state_dict(),
                'actor_target_state_dict': agents[aid].actor_target.state_dict(),
                'critic_state_dict': agents[aid].critic.state_dict(),
                'critic_target_state_dict': agents[aid].critic_target.state_dict(),
                'actor_optimizer_state_dict': agents[aid].actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agents[aid].critic_optimizer.state_dict(),
                # 'epsilon': agents[aid].epsilon,
                'steps_done': agents[aid].steps_done,
                'episode': episode + 1
            }
            torch.save(ckpt, os.path.join(cfg.checkpoint.path, f"{aid}_checkpoint_final.pt"))
            print(f"[Final] Checkpoint saved for {aid}")
            
    env.close()
    wandb.finish()
