import argparse
import os

import imageio
import torch
from omegaconf import OmegaConf
from pettingzoo.sisl import multiwalker_v9

from agents.ppo import PPOAgent


def generate_video(cfg, checkpoint_path):
    """Run a saved PPO policy and dump an MP4.

    Kompatybilne z nowym `ppo.py`, w którym:
    * sieć aktora to `agent.actor`
    * checkpointy zawierają klucz `"model_state_dict"` z wagami aktora
    * `act()` zwraca `(action, logp, value, u)` – bierzemy tylko pierwszy
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------- Env w trybie RGB-array ---------------------------------
    env = multiwalker_v9.parallel_env(render_mode="rgb_array")
    obs, _ = env.reset()

    obs_dim    = env.observation_space(env.possible_agents[0]).shape[0]
    action_dim = env.action_space(env.possible_agents[0]).shape[0]

    # ------------- Załaduj agenta + checkpoint ----------------------------
    agent = PPOAgent("shared", obs_dim, action_dim, device, cfg)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # ── nowy format (po zmianach w run()) ─────────────────────────────────
    if "actor_state_dict" in checkpoint:
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    # ── wsteczna kompatybilność (stary klucz) ─────────────────────────────
    else:
        agent.actor.load_state_dict(checkpoint["model_state_dict"], strict=False)

    agent.actor.eval()

    # ------------- Rollout & nagrywanie -----------------------------------
    frames = []
    done   = False

    while not done:
        # akcje dla wszystkich aktywnych agentów
        actions = {
            aid: agent.act(obs[aid])[0]  # pobierz tylko `action`
            for aid in env.agents if aid in obs
        }

        obs, _, terminations, truncations, _ = env.step(actions)
        done = not env.agents or any(terminations.values()) or any(truncations.values())

        frame = env.render()
        if frame is not None:
            frames.append(frame)

    # ------------- Zapis wideo --------------------------------------------
    os.makedirs("output", exist_ok=True)
    video_path = os.path.join("output", "video.mp4")
    imageio.mimsave(video_path, frames, fps=30)
    print(f"[✓] Video saved to {video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ppo.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_ppo_penalyty/shared_checkpoint_ep63500.pt", help="Path to model checkpoint")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    generate_video(cfg, args.checkpoint)
