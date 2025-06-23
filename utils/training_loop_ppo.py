import torch
import wandb
import numpy as np
from tqdm import trange
from pettingzoo.sisl import multiwalker_v9
from omegaconf import OmegaConf
import os
import random

from agents.ppo import PPOAgent
from utils.metrics import compute_distance, update_agent_effort, log_metrics_to_wandb
from utils.load_model import load_checkpoints
from utils.common import set_seed

def run(cfg):
    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=cfg.get("wandb_id", None),
        name=cfg.get("wandb_id", None),
        resume="allow"
    )
    print(f"[wandb] Logging to project: {cfg.project} (entity: {cfg.entity})")

    set_seed(cfg.seed)
    print(f"[seed] Random seed set to: {cfg.seed}")

    env = multiwalker_v9.parallel_env()
    action_space = env.action_space(env.possible_agents[0])
    print(f"Action space: {action_space}")
    print(f"Low: {action_space.low}")
    print(f"High: {action_space.high}")

    obs, _ = env.reset()
    print(f"[env] Loaded: multiwalker_v9 with {len(env.possible_agents)} agents")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")

    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]
    print(f"[obs] obs_dim: {obs_dim}, action_dim: {action_dim}")

    shared_agent = PPOAgent("shared", obs_dim, action_dim, device, cfg)

    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        print("[checkpoint] Attempting to load checkpoint...")
        ckpt_path = os.path.join(cfg.checkpoint.path, "shared_checkpoint_final.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            shared_agent.model.load_state_dict(ckpt['model_state_dict'])
            shared_agent.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_episode = ckpt.get('episode', 0)
            print(f"[resume] Loaded checkpoint, starting from episode {start_episode}")
        else:
            print(f"[resume] No checkpoint found, starting fresh")
            start_episode = 0
    else:
        start_episode = 0

    print(f"[agent] Shared agent initialized")

    total_steps = 0
    reward_history = []

    for episode in trange(start_episode, cfg.num_episodes, desc="Training"):
        obs, _ = env.reset()
        episode_reward = 0
        steps_in_episode = 0

        agent_efforts = {aid: 0.0 for aid in agent_ids}
        agent_rewards = {aid: 0.0 for aid in agent_ids}

        done = False
        while not done and steps_in_episode < cfg.max_steps:
            total_steps += 1
            steps_in_episode += 1

            actions = {
                aid: shared_agent.act(obs[aid])
                for aid in env.agents if aid in obs
            }

            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            dones = {
                aid: terminations.get(aid, False) or truncations.get(aid, False)
                for aid in agent_ids
            }

            for aid in actions.keys():
                if aid not in obs or aid not in next_obs:
                    continue

                agent_rewards[aid] += rewards.get(aid, 0.0)
                effort = np.sum(np.abs(actions[aid]))
                update_agent_effort(agent_efforts, aid, effort)
                shared_agent.store_reward(rewards[aid], dones[aid])

                # if "done_reason" in infos.get(aid, {}):
                #     print(f"[DEBUG DONE] {aid} done_reason: {infos[aid]['done_reason']}")
                # print(f"[DEBUG REWARD] {aid} reward: {rewards.get(aid, 0.0)}")

            obs = next_obs
            episode_reward += sum(rewards.values())

            if not env.agents:
                break

        reward_history.append(episode_reward)

        losses = shared_agent.learn()
        if losses is not None:
            episode_loss_critic, episode_loss_actor = losses
        else:
            episode_loss_critic = episode_loss_actor = 0.0

        avg_effort = np.mean(list(agent_efforts.values()))

        wandb.log({
            "episode": episode,
            "total_reward": episode_reward,
            "avg_critic_loss": episode_loss_critic,
            "avg_actor_loss": episode_loss_actor,
            "avg_effort": avg_effort,
            "steps_total": total_steps
        }, step=episode)

        if (episode + 1) % cfg.log_freq == 0:
            avg_reward = np.mean(reward_history[-cfg.log_freq:])
            print(f"Episode {episode} | Avg Reward: {avg_reward:.2f} | Critic Loss: {episode_loss_critic:.4f} | Actor Loss: {episode_loss_actor:.4f}")
            log_metrics_to_wandb(agent_efforts, agent_rewards, episode)

        if cfg.checkpoint.enabled and (episode + 1) % cfg.checkpoint.freq == 0:
            os.makedirs(cfg.checkpoint.path, exist_ok=True)
            torch.save({
                'model_state_dict': shared_agent.model.state_dict(),
                'optimizer_state_dict': shared_agent.optimizer.state_dict(),
                'episode': episode + 1
            }, os.path.join(cfg.checkpoint.path, f"shared_checkpoint_ep{episode+1}.pt"))
            print(f"Checkpoint saved at episode {episode+1}")

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
