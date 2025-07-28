# EffortSim

EffortSim is a research project exploring cooperation, laziness, and incentive structures in multi-agent reinforcement learning using the `multiwalker_v9` environment from PettingZoo.

---

### Features

- `generate_video.py`: Visualizes agent behavior — runs episodes and saves them to `.mp4` files

---

### Project Structure

- `agents/` — Agent logic and models  
  _(after many experiments, PPO gave the most stable results; not used agents like DDPG are stored in `archive/`)_

- `env/` — Environment wrappers and tracking tools

- `utils/` — Analytics, training loops, metrics, and model loading

- `archive/` — Older or experimental files preserved for review and reference

---

### Running the Project

```bash
conda env create -f environment.yaml
conda activate effortsim
python train.py
```

We provide a pretrained model and config file: [link here](https://ujchmura-my.sharepoint.com/:f:/g/personal/ignacy_kolton_student_uj_edu_pl/EjjTtcl_1bVPufUOSEaQJAkBIECwIbe4wR6ukuFQZsUVQw?e=5xxHYb)

in order to launch it put config file into `configs/` and weights into `checkpoints_ppo_penaty/`


Main training functionality is contained within files:
- train.py
- training_loop_ppo.py
- agents/ppo.y
- utils/metrics (we include penalization to encourage agen movement)

### Notes 
we use centralized PPO where all agents share weights. this was necessary because independent agents failed to converge.

at the moment model still converges to local minima and stops at some point we dont know why 

### TODO
- refactor the model loading mechanism in utils/load_model.py for shared model

- play with the reward system to avoid convergence to local optima
(see utils/metrics.py)

- make evaluation compatible with ppo (utils/eval_loop.py)

- add evaluation metrics to better track learning and cooperation
(utils/metrics.py)

- add per-agent effort estimation, measuring the work each agent puts into movement
  - Lazy Agent + Lazy Reward System

### This README is a work in progress and will evolve as the project develops.