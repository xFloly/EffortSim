"""
===============================================================================
PPO Multi-Agent Training Script for PettingZoo's Multiwalker Environment
===============================================================================

This script implements the full training loop for a multi-agent reinforcement
learning setup using the Proximal Policy Optimization (PPO) algorithm. The 
environment used is multiwalker_v9 from PettingZoo's sisl suite, which 
features multiple cooperative agents controlling different parts of a walker.

Key Components:

1. Environment Initialization:
   - Loads the parallel version of the multiwalker_v9 environment.
   - Configures environment-specific rewards for termination, falling, and forward movement.

2. Agent Setup:
   - A single shared PPO agent is instantiated to control all walkers, assuming a homogeneous agent design.
   - Observation and action dimensions are extracted from the environment.

3. Checkpointing:
   - If enabled in the configuration, the agent will resume training from the last saved checkpoint, including actor/critic model weights and optimizer state.

4. Training Loop:
   - Iterates over a set number of episodes.
   - At each step:
     - Agents select actions using the shared PPO policy.
     - Environment is stepped forward with those actions.
     - Rewards, done flags, and other metadata are collected.
     - Buffers per agent store observations, actions, log-probs, values, and rewards.

5. Policy Update:
   - After a fixed number of episodes, the agent performs a PPO update using the collected experience buffers.
   - Entropy coefficient (used to control exploration) is reduced periodically to encourage convergence.

6. Logging and Monitoring:
   - Uses Weights & Biases (wandb) to log episode rewards, losses, entropy coefficient, and other training metrics.
   - Outputs average performance statistics to the console every few episodes.

7. Saving Checkpoints:
   - Intermediate and final model states are saved for future resumption or evaluation.

"""

import os
import random
import numpy as np
import torch
import wandb
from tqdm import trange
from omegaconf import OmegaConf
from pettingzoo.sisl import multiwalker_v9
from utils.load_model import load_checkpoints
from agents.ppo import PPOAgent
from utils.metrics import compute_distance, update_agent_effort, log_metrics_to_wandb, penalty
# from utils.load_model import load_checkpoints
from utils.common import set_seed


def run(cfg, LAST_EVAL=False):
    """
    Main training loop for the PPO algorithm using the Multiwalker-v9 environment.
    """

    # ------------------------ WandB Logging Init ----------------------------
    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=cfg.get("wandb_id", None),
        name=cfg.get("wandb_id", None),
        resume="allow",
    )
    print(f"[wandb] Logging to project: {cfg.project} (entity: {cfg.entity})")

    # ------------------------ Random Seed -----------------------------------
    print(f"[seed] Random seed set to: {cfg.seed}")
    # Uncomment if reproducibility is needed:
    # set_seed(cfg.seed)

    # ------------------------ Environment Setup -----------------------------
    env = multiwalker_v9.parallel_env(
        terminate_reward=-100.0,
        fall_reward=-10.0,
        forward_reward=20.0,
    )
    print(f"[env] Loaded: multiwalker_v9 with {len(env.possible_agents)} agents")

    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]
    action_low = env.action_space(agent_ids[0]).low
    action_high = env.action_space(agent_ids[0]).high

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")
    print(f"[obs] obs_dim: {obs_dim}, action_dim: {action_dim}")

    # ------------------------ PPO Agent -------------------------------------
    shared_agent = PPOAgent("shared", obs_dim, action_dim, device, cfg)

    # ------------------------ Checkpoint Resume -----------------------------
    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        print("[checkpoint] Attempting to load checkpoint…")
        if load_checkpoints({"shared": shared_agent}, ["shared"], cfg):
            total_steps = shared_agent.steps_done
            start_episode = shared_agent.episode
            print(f"[resume] Loaded checkpoint at episode {start_episode}, total_steps={total_steps}")
        else:
            total_steps = 0
            start_episode = 0
            print("[resume] No checkpoint found – starting fresh")
    else:
        total_steps = 0
        start_episode = 0

    # ------------------------ Training Loop ---------------------------------
    reward_history = []
    episodes_since_last_train = 0

    for episode in trange(start_episode, cfg.num_episodes, desc="Training", initial=start_episode, total=cfg.num_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        steps_in_episode = 0

        # Initialize per-agent trackers
        agent_efforts = {aid: 0.0 for aid in agent_ids}
        agent_rewards = {aid: 0.0 for aid in agent_ids}
        prev_positions = {aid: (0.0, 0.0) for aid in agent_ids}

        # Initialize rollout buffers per agent
        agent_buffers = {
            aid: {
                "obs": [], "act": [], "u": [], "logp": [],
                "val": [], "rew": [], "done": [],
            } for aid in agent_ids
        }

        done = False
        while not done and steps_in_episode < cfg.max_steps:
            total_steps += 1
            steps_in_episode += 1

            # -------------------- Action Selection --------------------------
            actions = {}
            for aid in env.agents:
                if aid in obs:
                    action, logp, value, u = shared_agent.act(obs[aid])
                    buf = agent_buffers[aid]
                    buf["obs"].append(obs[aid])
                    buf["act"].append(action)
                    buf["u"].append(u)
                    buf["logp"].append(logp)
                    buf["val"].append(value)
                    actions[aid] = action  # Action in range [-1, 1]

            # -------------------- Environment Step --------------------------
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            dones = {
                aid: terminations.get(aid, False) or truncations.get(aid, False)
                for aid in agent_ids
            }

            for aid in env.agents:
                if aid not in agent_buffers:
                    continue

                base_reward = rewards.get(aid, 0.0)
                prev_pos = prev_positions.get(aid, (0.0, 0.0))
                current_pos = infos.get(aid, {}).get("walker_pos", prev_pos)

                r_penalty = penalty(current_pos, prev_pos)
                total_reward = base_reward + r_penalty

                buf = agent_buffers[aid]
                buf["rew"].append(total_reward)
                buf["done"].append(dones.get(aid, False))
                agent_rewards[aid] += total_reward

                prev_positions[aid] = current_pos

            obs = next_obs
            episode_reward += sum(rewards.values())

            if not env.agents:  # All agents are done
                break

        # -------------------- Store Buffers in Shared Agent -----------------
        for aid in agent_ids:
            buf = agent_buffers[aid]
            for o, a, u, l, r, d, v in zip(buf["obs"], buf["act"], buf["u"], buf["logp"],
                                          buf["rew"], buf["done"], buf["val"]):
                shared_agent.obs_buf.append(o)
                shared_agent.act_buf.append(a)
                shared_agent.u_buf.append(u)
                shared_agent.logp_buf.append(l)
                shared_agent.rew_buf.append(r)
                shared_agent.done_buf.append(d)
                shared_agent.val_buf.append(v)

        # -------------------- PPO Update ------------------------------------
        episodes_since_last_train += 1
        if episodes_since_last_train >= cfg.learn_every:
            losses = shared_agent.learn()
            shared_agent.reset_buffer()
            episodes_since_last_train = 0
        else:
            losses = None

        critic_loss, actor_loss = (losses if losses is not None else (0.0, 0.0))

        reward_history.append(episode_reward)
        shared_agent.episode += 1

        # -------------------- Entropy Annealing -----------------------------
        if (episode + 1) % 100 == 0:
            cfg.entropy_coef = max(cfg.entropy_coef * 0.9, 0.01)

        # -------------------- Logging ---------------------------------------
        wandb.log({
            "episode": episode,
            "total_reward": episode_reward,
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "steps_total": total_steps,
            "entropy_coef": cfg.entropy_coef,
        }, step=episode)

        if (episode + 1) % cfg.log_freq == 0:
            avg_reward = np.mean(reward_history[-cfg.log_freq:])
            print(
                f"Ep {episode}\tAvgReward {avg_reward:.1f}\tCritic {critic_loss:.3f}\tActor {actor_loss:.3f}"
            )
            log_metrics_to_wandb(agent_efforts, agent_rewards, episode)

        # -------------------- Checkpointing ---------------------------------
        if cfg.checkpoint.enabled and (episode + 1) % cfg.checkpoint.freq == 0:
            os.makedirs(cfg.checkpoint.path, exist_ok=True)
            ckpt_path = os.path.join(cfg.checkpoint.path, f"shared_checkpoint_ep{episode+1}.pt")
            torch.save({
                "actor_state_dict": shared_agent.actor.state_dict(),
                "critic_state_dict": shared_agent.critic.state_dict(),
                "optimizer_state_dict": shared_agent.optimizer.state_dict(),
                "episode": episode + 1,
                "steps_done": total_steps,
            }, ckpt_path)
            print(f"Checkpoint saved at episode {episode+1}")

    # ------------------------ Final Save ------------------------------------
    if cfg.checkpoint.enabled:
        os.makedirs(cfg.checkpoint.path, exist_ok=True)
        final_path = os.path.join(cfg.checkpoint.path, "shared_checkpoint_final.pt")
        torch.save({
            "actor_state_dict": shared_agent.actor.state_dict(),
            "critic_state_dict": shared_agent.critic.state_dict(),
            "optimizer_state_dict": shared_agent.optimizer.state_dict(),
            "episode": cfg.num_episodes,
            "steps_done": total_steps,
        }, final_path)
        print("[Final] Checkpoint saved")

    if LAST_EVAL:     
      eval_episodes = getattr(cfg, "eval_episodes", 10)
      eval_max_steps = getattr(cfg, "eval_max_steps", cfg.max_steps)
      final_eval = eval_after_training(cfg, shared_agent, num_episodes=eval_episodes, max_cycles=eval_max_steps)

    env.close()
    wandb.finish()

    return final_eval


def eval_after_training(cfg, agent, num_episodes=10, max_cycles=500):
    """
    Returns: mean reward per walker per episode (float)
    Uses the same reward accounting as your evaluate() for PPO.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_reward = 0.0
    total_agents = 0
    # create a fresh env per episode (no rendering for speed)
    for _ in range(num_episodes):
        env = multiwalker_v9.parallel_env(
            terminate_reward=-100.0,
            fall_reward=-10.0,
            forward_reward=20.0,
        )
        obs, _ = env.reset()
        agent_ids = env.possible_agents
        ep_reward_sum = {aid: 0.0 for aid in agent_ids}

        for _ in range(max_cycles):
            actions = {
                aid: agent.act(obs[aid])[0]
                for aid in env.agents
                if aid in obs
            }
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            for aid in env.agents:
                if aid in rewards:
                    ep_reward_sum[aid] += rewards[aid]

            obs = next_obs
            done = (not env.agents) or (all(terminations.values()) or all(truncations.values()))
            if done:
                break

        total_reward += sum(ep_reward_sum.values())
        total_agents = len(agent_ids) 
        env.close()

    # average per walker per episode
    return float(total_reward / (num_episodes * total_agents))
