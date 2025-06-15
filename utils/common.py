
import random
import torch
import wandb
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if wandb.run is not None:
            wandb.config.update({"seed": seed})
    return