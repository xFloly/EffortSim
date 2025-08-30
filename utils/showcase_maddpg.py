import time
import torch
import numpy as np
from pettingzoo.sisl import multiwalker_v9

from agents.maddpg import MADDPG
from utils.maddpg_io import load_maddpg_checkpoints
from utils.common import set_seed


def showcase(cfg, max_cycles=500, sleep_s=0.05, noise_std=0.0):
    """
    Visualize a trained MADDPG policy in the multiwalker_v9 env.

    Args:
        cfg: your Hydra/OmegaConf config with .checkpoint fields
        max_cycles: max env steps per episode
        sleep_s: wall-clock delay between renders
        noise_std: exploration noise (0.0 recommended for showcase)
    """
    # Seed & device
    set_seed(cfg.seed)
    print(f"[seed] Random seed set to: {cfg.seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Env with rendering
    env = multiwalker_v9.parallel_env(render_mode="human",        
                                      terminate_reward=-100.0,
                                      fall_reward=-10.0,
                                      forward_reward=20.0)
    env.reset()

    # Agent metadata
    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]

    # MADDPG (central critic, decentralized actors)
    maddpg = MADDPG(agent_ids, obs_dim, action_dim, device, cfg)

    # Load checkpoints (if configured)
    if cfg.checkpoint.enabled and cfg.checkpoint.resume:
        loaded_any, total_steps, start_episode = load_maddpg_checkpoints(maddpg, agent_ids, cfg)
        if loaded_any:
            print(f"[showcase] Loaded MADDPG checkpoints. start_episode={start_episode}, total_steps={total_steps}")
        else:
            print("[showcase] No MADDPG checkpoints found; running with fresh weights.")

    print("\n[showcase] Running MADDPG showcase loop continuously.")
    episode = 0

    try:
        while True:
            episode += 1
            print(f"\nShowcase Episode {episode}")
            obs, _ = env.reset()
            total_reward_episode = 0.0

            for step in range(max_cycles):
                env.render()
                time.sleep(sleep_s)

                # Act with *no noise* by default for clean showcase
                # maddpg.act expects a dict of obs; returns dict of actions
                actions = {}
                if env.agents:
                    # build action dict only for active agents present in obs
                    obs_active = {aid: obs[aid] for aid in env.agents if aid in obs}
                    if obs_active:
                        actions = maddpg.act(obs_active, noise_std=noise_std)

                if not actions:
                    # nothing to step (episode likely over)
                    break

                next_obs, rewards, terminations, truncations, infos = env.step(actions)

                # Termination handling similar to your DDPG showcase
                dones = terminations  # PettingZoo v9 naming
                total_reward_episode += float(sum(rewards.values()))

                # If all agents terminated, stop episode
                if dones and all(dones.values()):
                    obs = next_obs
                    break

                obs = next_obs

            print(f"End of Showcase Episode {episode}. Total Reward: {total_reward_episode:.2f}")

    except KeyboardInterrupt:
        print("\n[showcase] Showcase interrupted by user.")

    env.close()
