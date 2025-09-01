# EffortSim
## Ignacy Kolton, Kacper Marzol, Filip Soszyński
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

[//]: # (### Notes )

[//]: # (we use centralized PPO where all agents share weights. this was necessary because independent agents failed to converge.)

[//]: # ()
[//]: # (at the moment model still converges to local minima and stops at some point we dont know why )


## Algorithm & Model Details

## PPO
Agent setup: One shared policy network for all walkers; agents share weights to ensure stable learning. 

Network architecture: Fully connected MLP, 2 hidden layers of 128 units each, ReLU activations.

Hyperparameters:
* Learning rate: 0.0003 
* Clip ratio: 0.2 
* Batch size: 128

## DDPG 
### Shared
**Agent setup**: Single actor-critic network shared among all walkers.
**Architecture**: Actor and critic both MLPs with 2 hidden layers of 128 units, ReLU.

### Independent
**Agent setup**: Each walker has its own actor-critic network.
**Architecture**: Same as above, but independent for each agent.

### Multi-Agent
**Agent setup**: Each walker has its own actor network but shares a common critic.
**Architecture**: Actor: 2 hidden layers, 128 units, ReLU; Critic: 2 hidden layers, 128 units, ReLU.


## Reward & Incentive Design
The reward structure in EffortSim is designed to balance cooperation with individual effort, encouraging agents to move efficiently while maintaining collective success.

### Forward Progress
Agents receive positive reward proportional to their forward movement.
Encourages walkers to move together rather than stop. 
### Lazy-Agent Penalty
Penalizes agents who contribute minimal movement or remain idle.

# Results and experiments
We compare two models with different approaches, we provide movies created after the training to compare the cooperation 
## PPO

![Video preview](readme/ppo.gif)


## DDPG
### Shared
One universal model for all 3 walkers

<video width="300" controls>
  <source src="readme/ddpg_shared_run.mp4" type="video/mp4">
</video>

### Independent
Each walker has its own network

<video width="300" controls>
  <source src="readme/ddpg_independet_run.mp4" type="video/mp4">
</video>

### Multi-agent 
Each walker has its own agent, but they share the critic

<video width="300" controls>
  <source src="readme/ddpg_ma_run.mp4" type="video/mp4">
</video>

The results are summarized in the table 

Algorithm | Average Reward | Cooperation Rate | Notes
--- |----------------| --- | ---
DDPG_shared| *:(*           | *:(* | ?
DDPG_independent | *:(*     | *:(* | Unstable
DDPG_multiagent |*:(*     | *:(* | Failed to converge
PPO | *:(*      | *:(* |  prone to local minima

We also provide plots showcasing the training process:

![plots](readme/plot.png)



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