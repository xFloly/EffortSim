import os
import random
import numpy as np
import torch
import wandb
from tqdm import trange
from omegaconf import OmegaConf
from pettingzoo.sisl import multiwalker_v9

from agents.ppo import PPOAgent
from utils.metrics import compute_distance, update_agent_effort, log_metrics_to_wandb, penalty
from utils.load_model import load_checkpoints
from utils.common import set_seed


# -----------------------------------------------------------------------------
#  Main training loop – now kompatybilny z nowym ppo.py (unsquashed-Gaussian)
# -----------------------------------------------------------------------------

def run(cfg):
    """
    Main training loop for PPO on Multiwalker-v9.
    
    Główny workflow eksperymentu: inicjalizacja, pętla PPO, logowanie.

    Zakładamy, że **ppo.py** zwraca teraz `(action, logp, value, u)`.
    Funkcja została przerobiona tak, żeby zapisywać również `u` w buforach.
    """

    # ------------------------ WandB -----------------------------------------
    wandb.init(
        project=cfg.project,
        entity=cfg.entity,
        config=OmegaConf.to_container(cfg, resolve=True),
        id=cfg.get("wandb_id", None),
        name=cfg.get("wandb_id", None),
        resume="allow",
    )
    print(f"[wandb] Logging to project: {cfg.project} (entity: {cfg.entity})")

    # ------------------------ SEED -----------------------------------------
    # set_seed(cfg.seed)
    print(f"[seed] Random seed set to: {cfg.seed}")

    # ------------------------ ENV ------------------------------------------
    env = multiwalker_v9.parallel_env(
        # shared_reward=False,
        terminate_reward=-100.0,
        fall_reward=-10.0,
        forward_reward=20.0,
    )
    # env = multiwalker_v9.parallel_env()
    print(f"[env] Loaded: multiwalker_v9 with {len(env.possible_agents)} agents")

    agent_ids   = env.possible_agents
    obs_dim     = env.observation_space(agent_ids[0]).shape[0]
    action_dim  = env.action_space(agent_ids[0]).shape[0]
    action_low  = env.action_space(agent_ids[0]).low
    action_high = env.action_space(agent_ids[0]).high

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] Using: {device}")
    print(f"[obs] obs_dim: {obs_dim}, action_dim: {action_dim}")

    # ------------------------ Agent ----------------------------------------
    shared_agent = PPOAgent("shared", obs_dim, action_dim, device, cfg)

    # ------------------------ Checkpoint resume ----------------------------
    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        print("[checkpoint] Attempting to load checkpoint…")
        if load_checkpoints({"shared": shared_agent}, ["shared"], cfg):
            total_steps   = shared_agent.steps_done
            start_episode = shared_agent.episode
            print(f"[resume] Loaded checkpoint at episode {start_episode}, total_steps={total_steps}")
        else:
            total_steps = 0
            start_episode = 0
            print("[resume] No checkpoint found – starting fresh")
    else:
        total_steps = 0
        start_episode = 0

    # ------------------------ Training loop --------------------------------
    reward_history = []
    episodes_since_last_train = 0

    for episode in trange(start_episode, cfg.num_episodes, desc="Training", initial=start_episode, total=cfg.num_episodes):
        obs, _ = env.reset()
        episode_reward   = 0.0
        steps_in_episode = 0

        # per-agent helpers (effort, reward, prev-pos etc.)
        agent_efforts  = {aid: 0.0 for aid in agent_ids}
        agent_rewards  = {aid: 0.0 for aid in agent_ids}
        prev_positions = {aid: (0.0, 0.0) for aid in agent_ids}

        # rollout buffers per agent (obs, act, u, …)
        agent_buffers = {
            aid: {
                "obs":  [],
                "act":  [],
                "u":    [],   # <── NEW
                "logp": [],
                "val":  [],
                "rew":  [],
                "done": [],
            }
            for aid in agent_ids
        }

        done = False
        while not done and steps_in_episode < cfg.max_steps:
            total_steps   += 1
            steps_in_episode += 1

            # --------------- Action selection ------------------------------
            actions = {}
            for aid in env.agents:
                if aid in obs:
                    action, logp, value, u = shared_agent.act(obs[aid])

                    # zapisz do local buffer
                    buf = agent_buffers[aid]
                    buf["obs" ].append(obs[aid])
                    buf["act" ].append(action)
                    buf["u"   ].append(u)      # <── NEW
                    buf["logp"].append(logp)
                    buf["val" ].append(value)

                    actions[aid] = action  # numpy z [-1,1]

            # --------------- Env step -------------------------------------
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            dones = {
                aid: terminations.get(aid, False) or truncations.get(aid, False)
                for aid in agent_ids
            }

            for aid in env.agents:
                if aid not in agent_buffers:
                    continue

                base_r = rewards.get(aid, 0.0)
                prev   = prev_positions.get(aid, (0.0, 0.0))
                curr   = infos.get(aid, {}).get("walker_pos", prev)

                r_penalty = penalty(curr, prev)
                total_r   = base_r + r_penalty

                buf = agent_buffers[aid]
                buf["rew" ].append(total_r)
                buf["done"].append(dones.get(aid, False))
                agent_rewards[aid] += total_r

                # update helper tracking
                prev_positions[aid] = curr

            obs = next_obs
            episode_reward += sum(rewards.values())

            if not env.agents:  # all agents out
                break

        # ----------------- Copy per-agent buffers → shared ------------------
        for aid in agent_ids:
            buf = agent_buffers[aid]
            for o, a, u, l, r, d, v in zip(
                buf["obs"], buf["act"], buf["u"], buf["logp"],
                buf["rew"], buf["done"], buf["val"]
            ):
                shared_agent.obs_buf.append(o)
                shared_agent.act_buf.append(a)
                shared_agent.u_buf.append(u)      # <── NEW
                shared_agent.logp_buf.append(l)
                shared_agent.rew_buf.append(r)
                shared_agent.done_buf.append(d)
                shared_agent.val_buf.append(v)

        # ----------------- PPO update --------------------------------------
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

        # # ----------------- Entropy annealing -------------------------------
        if (episode + 1) % 100 == 0:
            cfg.entropy_coef = max(cfg.entropy_coef * 0.9, 0.01)

        # ----------------- Logging ----------------------------------------
        wandb.log(
            {
                "episode": episode,
                "total_reward": episode_reward,
                "critic_loss": critic_loss,
                "actor_loss": actor_loss,
                "steps_total": total_steps,
                "entropy_coef": cfg.entropy_coef,

            },
            step=episode,
        )

        if (episode + 1) % cfg.log_freq == 0:
            avg_reward = np.mean(reward_history[-cfg.log_freq:])
            print(
                f"Ep {episode}\tAvgReward {avg_reward:.1f}\tCritic {critic_loss:.3f}\tActor {actor_loss:.3f}"
            )
            log_metrics_to_wandb(agent_efforts, agent_rewards, episode)

        # ----------------- Checkpoint -------------------------------------
        if cfg.checkpoint.enabled and (episode + 1) % cfg.checkpoint.freq == 0:
            os.makedirs(cfg.checkpoint.path, exist_ok=True)

            ckpt_path = os.path.join(
                cfg.checkpoint.path, f"shared_checkpoint_ep{episode+1}.pt"
            )
            torch.save(
                {
                    "actor_state_dict":  shared_agent.actor.state_dict(),
                    "critic_state_dict": shared_agent.critic.state_dict(),
                    "optimizer_state_dict": shared_agent.optimizer.state_dict(),
                    "episode":     episode + 1,
                    "steps_done":  total_steps,
                },
                ckpt_path,
            )
            print(f"Checkpoint saved at episode {episode+1}")

    # ------------------------ Final save -----------------------------------
    if cfg.checkpoint.enabled:
        os.makedirs(cfg.checkpoint.path, exist_ok=True)

        final_path = os.path.join(cfg.checkpoint.path, "shared_checkpoint_final.pt")
        torch.save(
            {
                "actor_state_dict":  shared_agent.actor.state_dict(),
                "critic_state_dict": shared_agent.critic.state_dict(),
                "optimizer_state_dict": shared_agent.optimizer.state_dict(),
                "episode":     cfg.num_episodes,
                "steps_done":  total_steps,
            },
            final_path,
        )
        print("[Final] Checkpoint saved")

    env.close()
    wandb.finish()
