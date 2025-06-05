# export OMP_NUM_THREADS=1
# export KMP_DUPLICATE_LIB_OK=TRUE
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import argparse
from omegaconf import OmegaConf
from utils.training_loop import run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    run(cfg)

if __name__ == "__main__":
    main()
