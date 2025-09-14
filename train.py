import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
from omegaconf import OmegaConf
from policy import ppo, ddpg_independent, ddpg_shared, maddpg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="configs/default.yaml")
    parser.add_argument("--algo", choices=["ppo", "ddpg_independent", "ddpg_shared", "maddpg"], default="ppo", help="Algorithm to train")

    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if args.algo == "ppo":
        ppo.run(cfg)
    elif args.algo == "ddpg_independent":
        ddpg_independent.run(cfg)
    elif args.algo == "ddpg_shared":
        ddpg_shared.run(cfg)
    elif args.algo == "maddpg":
        maddpg.run(cfg)
    else:
        raise NotImplementedError(f"Algorithm {args.algo} not yet implemented.")

if __name__ == "__main__":
    main()

# python train.py --config path_to_config --algo ppo
