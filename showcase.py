import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
from omegaconf import OmegaConf
from utils.showcase_loop import showcase


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="configs/default.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    showcase(cfg) #showcase_loop


if __name__ == "__main__":
    main()