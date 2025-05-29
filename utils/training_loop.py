import torch
import wandb
import numpy as np
from tqdm import trange
from pettingzoo.sisl import multiwalker_v9
from omegaconf import OmegaConf
import os
import random

from agents.dqn_agent import DQNAgent
from utils.metrics import compute_distance, update_agent_effort, log_metrics_to_wandb
from utils.load_model import load_checkpoints

def run(cfg):
    ### WandB initialization ###
    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    print(f"[wandb] Logging to project: {cfg.project} (entity: {cfg.entity})")
    
    ### Set SEED for reproducibility ###
    seed = cfg.get("seed", 42)  # fallback to 42 if not specified
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[seed] Random seed set to: {seed}")

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
        aid: DQNAgent(aid, obs_dim, action_dim, device, cfg)
        for aid in agent_ids
    }
    print(f"[agents] Initialized: {', '.join(agent_ids)}")

    ### Load pre-trained model checkpoints if resume is enabled ###
    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        load_checkpoints(agents, agent_ids, cfg)
    
    ### Train Loop
    reward_history = []
    total_steps = 0

    for episode in trange(cfg.num_episodes, desc="Training"):
        obs, _ = env.reset()
        episode_reward = 0
        episode_loss = 0.0
        steps_in_episode = 0

        agent_efforts = {aid: 0.0 for aid in agent_ids}
        agent_rewards = {aid: 0.0 for aid in agent_ids}
        prev_positions = {aid: (0.0, 0.0) for aid in agent_ids}

        for step in range(cfg.max_steps):
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
            step_loss = 0
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
                loss = agents[aid].learn()
                if loss is not None:
                    step_loss += loss
                    active_agents += 1

            obs = next_obs
            episode_reward += sum(rewards.values())
            if active_agents > 0:
                episode_loss += step_loss / active_agents

            # Break if all agents terminated 
            if not env.agents:
                break

        avg_loss = episode_loss / steps_in_episode if steps_in_episode > 0 else 0
        reward_history.append(episode_reward)
 
        #after `target_update_freq` episodes update the agents:
        if (episode + 1) % cfg.target_update_freq == 0:
            for agent in agents.values():
                agent.update_target_net()

        ### LOG to wandb ###
        wandb.log({
            "episode": episode,
            "total_reward": episode_reward,
            "avg_loss": avg_loss,
            "epsilon": agents[agent_ids[0]].epsilon,
            "steps_total": total_steps
        }, step=episode)

        log_metrics_to_wandb(agent_efforts, agent_rewards, step=episode)

        ### after log_freq episodes : Print LOGS ###
        if (episode + 1) % cfg.log_freq == 0:
            avg20 = np.mean(reward_history[-20:]) if len(reward_history) >= 20 else np.mean(reward_history)
            print(f"Ep {episode+1}/{cfg.num_episodes} | Reward: {episode_reward:.2f} | Avg(20): {avg20:.2f} | Loss: {avg_loss:.4f} | Eps: {agents[agent_ids[0]].epsilon:.3f}")

        ### after checkpoint_freq episodes : Add Checkpoint ###
        if cfg.checkpoint.enabled and (episode + 1) % cfg.checkpoint_freq == 0:
            os.makedirs(cfg.checkpoint.path, exist_ok=True)
            for aid in agent_ids:
                ckpt = {
                    'policy_state_dict': agents[aid].policy_net.state_dict(),
                    'target_state_dict': agents[aid].target_net.state_dict(),
                    'optimizer_state_dict': agents[aid].optimizer.state_dict(),
                    'epsilon': agents[aid].epsilon,
                    'steps_done': agents[aid].steps_done
                }
                torch.save(ckpt, os.path.join(cfg.checkpoint.path, f"{aid}_checkpoint.pt"))
                print(f"Checkpoint saved for {aid}")


    ### Saving model after training###
    if cfg.checkpoint.enabled:
        os.makedirs(cfg.checkpoint.path, exist_ok=True)
        for aid in agent_ids:
            ckpt = {
                'policy_state_dict': agents[aid].policy_net.state_dict(),
                'target_state_dict': agents[aid].target_net.state_dict(),
                'optimizer_state_dict': agents[aid].optimizer.state_dict(),
                'epsilon': agents[aid].epsilon,
                'steps_done': agents[aid].steps_done
            }
            torch.save(ckpt, os.path.join(cfg.checkpoint.path, f"{aid}_checkpoint_final.pt"))
            print(f"[final] Checkpoint saved for {aid}")
            
    env.close()
    wandb.finish()
