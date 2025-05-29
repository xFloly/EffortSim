import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
from omegaconf import OmegaConf
from utils.eval_loop import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="configs/default.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    evaluate(cfg)


if __name__ == "__main__":
    main()


#python evaluate.py --config path_to_config