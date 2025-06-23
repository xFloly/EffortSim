import torch
import argparse
import imageio
import os
from omegaconf import OmegaConf
from pettingzoo.sisl import multiwalker_v9

from agents.ppo import PPOAgent


def generate_video(cfg, checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = multiwalker_v9.parallel_env(render_mode="rgb_array")
    obs, _ = env.reset()

    obs_dim = env.observation_space(env.possible_agents[0]).shape[0]
    action_dim = env.action_space(env.possible_agents[0]).shape[0]

    agent = PPOAgent("shared", obs_dim, action_dim, device, cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    agent.model.load_state_dict(checkpoint["model_state_dict"])
    agent.model.eval()

    frames = []
    done = False

    while not done:
        actions = {
            aid: agent.act(obs[aid])
            for aid in env.agents if aid in obs
        }
        obs, _, terminations, truncations, _ = env.step(actions)

        frame = env.render()
        frames.append(frame)

        done = not env.agents

    os.makedirs("output", exist_ok=True)
    video_path = "output/video.mp4"
    imageio.mimsave(video_path, frames, fps=30)
    print(f"[✓] Video saved to {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/shared_checkpoint_ep1000.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    generate_video(cfg, args.checkpoint)
