import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
from omegaconf import OmegaConf
from utils.eval_loop import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml")
    parser.add_argument("--agent", type=str, choices=["ppo", "ddpg", "maddpg"])
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    if args.agent is not None:
        cfg.agent = args.agent
    if args.model_path is not None:
        if "checkpoint" not in cfg or cfg.checkpoint is None:
            cfg.checkpoint = {}
        cfg.checkpoint.path = args.model_path

    evaluate(cfg)
if __name__ == "__main__":
    main()

# python evaluate.py --config path/to/your_config.yaml --agent maddpg --model_path path/to/your_model.pt
