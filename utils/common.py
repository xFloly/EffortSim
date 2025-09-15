def set_seed(cfg, env=None):
    """
    Set random seed for Python, NumPy, PyTorch, and optionally a PettingZoo/Gym env.
    """
    import random
    import torch
    import numpy as np

    seed = cfg.seed
    env_seed = getattr(cfg, "env_seed", seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if env is not None:
        try:
            env.reset(seed=env_seed)
        except TypeError:
            env.seed(env_seed)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(env_seed)
        if hasattr(env.observation_space, "seed"):
            env.observation_space.seed(env_seed)

    print(f"[seed] Global seed set to {seed}")
    print(f"[seed] Env seed set to {env_seed}")
