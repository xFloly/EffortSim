import warnings
warnings.filterwarnings("ignore")

import time
import torch
import numpy as np
import imageio
from pettingzoo.sisl import multiwalker_v9
from utils.common import set_seed
from utils.load_model import load_agent, load_checkpoints, load_maddpg_checkpoints
from agents.maddpg import MADDPG
from agents.ddpg import DDPGAgent

def generate_video(cfg, agent_name, model_path, output_path, max_cycles=500, sleep_s=0.0):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] Using device: {device}, seed: {cfg.seed}")

    env = multiwalker_v9.parallel_env(render_mode="rgb_array",
                                      terminate_reward=-100.0,
                                      fall_reward=-10.0,
                                      forward_reward=20.0)
    obs, _ = env.reset()
    agent_name = agent_name.lower()
    multi_agent = False
    frames = []



    # if agent_name == "maddpg":
    #     agent = load_agent("maddpg", model_path, env, device = device, cfg = cfg)
    #     multi_agent = True
    # elif agent_name == "ddpg":
    #     agent = load_agent("ddpg", model_path, env, device = device, cfg = cfg)
    # elif agent_name == "ppo":
    #     agent = load_agent("ppo", model_path, env, device = device, cfg = cfg)
    # else:
    #     raise ValueError(f"Unsupported agent: {agent_name}")

    agent_ids = env.possible_agents
    obs_space = env.observation_space(agent_ids[0])
    obs_dim = obs_space.shape[0]  # if Box space
    action_space = env.action_space(agent_ids[0])
    action_dim = action_space.shape[0]

    if agent_name == "maddpg":
        from agents.maddpg import MADDPG
        agent = MADDPG(agent_ids, obs_dim, action_dim, device=device, cfg=cfg)
        multi_agent = True
        load_maddpg_checkpoints(agent, agent_ids, cfg, model_path)
    elif agent_name == "ddpg":
        from agents.multi_ddpg import MultiDDPG
        agent = MultiDDPG(agent_ids, obs_dim, action_dim, device=device, cfg=cfg, checkpoint_path=model_path)
        multi_agent = True
    elif agent_name == "ppo":
        agent = load_agent("ppo", model_path, env, device=device, cfg=cfg)
    else:
        raise ValueError(f"Unsupported agent: {agent_name}")

    total_reward_episode = 0
    for step in range(max_cycles):
        if multi_agent:
            actions = agent.act(obs, noise_std=0.0)
        else:
            actions = {aid: agent.act(obs[aid])[0] for aid in env.agents if aid in obs}

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        total_reward_episode += sum(rewards.values())
        obs = next_obs

        done = not env.agents or any(terminations.values()) or any(truncations.values())
        if done:
            break

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imageio.mimsave(output_path, frames, fps=30)
    print(f"[eval] Video saved at: {output_path}")
    print(f"[eval] Total reward {total_reward_episode}")

    env.close()
    return total_reward_episode


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["ppo", "ddpg", "maddpg"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output", type=str, default="outputs/videos/demo.mp4")
    parser.add_argument("--max_cycles", type=int, default=500)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config) if args.config else None
    generate_video(cfg, args.agent, args.model_path, args.output, args.max_cycles, args.sleep)


# python generate_video.py --agent ddpg --model_path checkpoints/ddpg_final.pt --output outputs/videos/ddpg_demo.mp4


# python generate_video.py --config Social/configs/ddpg_shared.yaml --agent ddpg --model_path Social/checkpoint_ddpg_penalty --output outputs/videos/ddpg_shared.mp4
#  python generate_video.py --config Social/configs/ddpg_individual.yaml --agent ddpg --model_path Social/checkpoint_ddpg_independent --output outputs/videos/ddpg_individual.mp4
#  python generate_video.py --config Social/configs/ppo.yaml --agent ppo --model_path Social/checkpoints/ppo_trial_30_best_hypertuning/shared_checkpoint_ep4000.pt --output outputs/videos/ppo.mp4
# python generate_video.py --config Social/configs/mmdpg.yaml --agent maddpg --model_path Social/checkpoint_maddpg --output outputs/videos/maddpg.mp4
