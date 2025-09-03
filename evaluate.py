import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
from omegaconf import OmegaConf
from utils.eval_loop import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, choices=["ppo", "ddpg", "maddpg"], required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max_cycles", type=int, default=500)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    evaluate(cfg, args.agent, args.model_path, num_episodes=args.episodes, max_cycles=args.max_cycles)

if __name__ == "__main__":
    main()

# python evaluate.py --config path/to/your_config.yaml --agent maddpg --model_path path/to/your_model.pt
