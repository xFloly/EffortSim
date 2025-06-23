import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
from omegaconf import OmegaConf
from utils.training_loop_ppo import run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="configs/ppo.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    run(cfg) #training_loop


if __name__ == "__main__":
    main()

#python train.py --config path_to_config
