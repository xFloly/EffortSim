# EffortSim

EffortSim is a research project exploring cooperation, laziness, and incentive structures in multi-agent reinforcement learning using the `multiwalker_v9` environment from PettingZoo.

### Features:

- evaluate : runs episodes on evaluation with metrics 
- showcase: runs episodes with visible game


### Structure:
- `agents/` — agent logic and models
- `env/` — environment wrappers and tracking tools
- `utils/` — analytics and reporting tools

### To run:
```bash
conda env create -f environment.yaml
conda activate effortsim
python train.py
```

### TODO
- metrics for training - rewards (utils/metrics.py)

- metrics for evaluation (utils/metrics.py)

- poprawic trainig loopa zeby nie robił (utils/training_loop.py) unlearningu

- classical DQN Agent (agents/.)

- dodać id runów do wandb (utils/training_loop.py) #DONE

- Lazy Agent + Lazy Reward System (metrics/training_loop)

- add loading from shared file (utils/load_model) 