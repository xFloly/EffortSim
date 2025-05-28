# EffortSim

EffortSim is a research project exploring cooperation, laziness, and incentive structures in multi-agent reinforcement learning using the `multiwalker_v9` environment from PettingZoo.

### Features:
- DQN agents implemented using TorchRL
- Track and analyze individual agent contributions
- Integrated with Weights & Biases for logging
- Modular project layout for experimentation

### Structure:
- `agents/` — agent logic and models
- `env/` — environment wrappers and tracking tools
- `scripts/` — training and evaluation scripts
- `utils/` — analytics and reporting tools

### To run:
```bash
conda env create -f environment.yaml
conda activate effortsim
python train.py
```