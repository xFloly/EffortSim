import torch
import time
from pettingzoo.sisl import multiwalker_v9
from agents.ddpg import DDPGAgent  # or wherever your agent class is
from omegaconf import OmegaConf
import argparse


def load_agents(cfg, checkpoint_path, env, device):
    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]

    agents = {}
    for aid in agent_ids:
        agent = DDPGAgent(aid, obs_dim, action_dim, device, cfg)
        checkpoint_file = f"{checkpoint_path}/{aid}_checkpoint_ep2000.pt"
        state = torch.load(checkpoint_file, map_location=device)
        agent.actor.load_state_dict(state['actor_state_dict'])
        agent.actor.eval()
        agents[aid] = agent

    return agents


def visualize(cfg, checkpoint_path):
    env = multiwalker_v9.parallel_env(render_mode="human")
    obs, _ = env.reset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agents = load_agents(cfg, checkpoint_path, env, device)

    done = {aid: False for aid in env.possible_agents}
    frames_per_second = 120
    frame_delay = 1.0 / frames_per_second

    while not all(done.values()):
        actions = {
            aid: agents[aid].act(obs[aid], noise=False)
            for aid in env.agents if not done[aid]
        }

        obs, rewards, terminations, truncations, infos = env.step(actions)

        done = {
            aid: terminations.get(aid, False) or truncations.get(aid, False)
            for aid in env.agents
        }

        time.sleep(frame_delay)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    visualize(cfg, args.checkpoint_path)
