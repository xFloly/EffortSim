# EffortSim

**EffortSim** is a coursework research project exploring cooperation, laziness, and incentive structures in multi‑agent reinforcement learning using the `multiwalker_v9` environment from PettingZoo.

## Authors

Ignacy Kolton, Kacper Marzol, Filip Soszyński

---

## Features

* `train.py` — launch **IPPO** (parameter‑sharing) training on `multiwalker_v9`.
* `generate_video.py` — run episodes with a trained policy and export `.mp4`.
* `policy/ddpg_independent.py`, `policy/ddpg_shared.py`, `policy/maddpg.py` — optional **IDDPG/MADDPG** baselines.
* `utils/metrics.py` — simple cooperation/effort metrics for shaping and logging.
* `utils/load_model.py` / `utils/maddpg_io.py` — checkpoint I/O utilities.

> Note: after many experiments, PPO/IPPO gave the most stable results; unused/older code (e.g., DQN) is kept in `archive/`.

---

## Project Structure

```text
EffortSim/
├── agents
│   ├── ddpg.py
│   ├── maddpg.py
│   └── ppo.py
├── archive
│   └── agents
│       └── dqn_agent.py
├── configs
│   ├── ddpg.yaml
│   ├── default.yaml
│   └── ppo.yaml
├── policy
│   ├── ddpg_independent.py
│   ├── ddpg_shared.py
│   └── maddpg.py
├── utils
│   ├── common.py
│   ├── eval_loop.py
│   ├── load_model.py
│   ├── maddpg_io.py
│   ├── metrics.py
│   ├── showcase_loop.py
│   ├── showcase_maddpg.py
│   └── training_loop_ppo.py
├── environment.yaml
├── environment_no_cuda.yaml
├── evaluate.py
├── generate_video.py
├── LICENSE
├── README.md
├── showcase.py
└── train.py
```

---

## Installation

CUDA‑enabled:

```bash
conda env create -f environment.yaml
conda activate effortsim
```

CPU/macOS:

```bash
conda env create -f environment_no_cuda.yaml
conda activate effortsim
```

---

## Running IPPO (Parameter Sharing)

```bash
python train.py --config configs/ppo.yaml
# or
python train.py --config configs/default.yaml
```

To export a video from a checkpoint:

```bash
python generate_video.py --config configs/ppo.yaml \
  --checkpoint checkpoints/shared_checkpoint_epXXXX.pt
```

Optional baselines:

```bash
# IDDPG (independent)
python -c "from policy.ddpg_independent import run; \
from omegaconf import OmegaConf; run(OmegaConf.load('configs/ddpg.yaml'))"

# DDPG with parameter sharing
python -c "from policy.ddpg_shared import run; \
from omegaconf import OmegaConf; run(OmegaConf.load('configs/ddpg.yaml'))"

# MADDPG (centralized critic)
python -c "from policy.maddpg import run; \
from omegaconf import OmegaConf; run(OmegaConf.load('configs/ddpg.yaml'))"
```

---

## Terminology

PPO and DDPG are single‑agent algorithms. In this repository we apply them in a decentralized multi‑agent way:

* **IPPO**: Independent PPO; we use **parameter sharing** (one policy for all agents).
* **IDDPG**: Independent DDPG (independent and parameter‑sharing variants).
* **MADDPG**: centralized critic with per‑agent actors.

We keep file names (`ppo.py`, `ddpg.py`) for continuity and refer to the multi‑agent formulations as IPPO/IDDPG in this document.

---

## Algorithms

**IPPO (parameter sharing).** A single PPO policy and value function are shared across all walkers. Each agent supplies its own observation; actions are sampled from a tanh‑squashed Gaussian. The training loop (`utils/training_loop_ppo.py`) collects per‑agent rollouts and applies the clipped PPO objective with GAE. Default settings: `eps_clip=0.2`, `entropy_coef=0.01`, `value_loss_coef=0.5`, `ppo_epochs=15`, `mini_batch_size=128`, gradient clip `0.5` (see `configs/ppo.yaml`).

**IDDPG.** Each agent has an actor μ(o) and critic Q(o,a) with replay and target networks (`agents/ddpg.py`). Two runners are provided: independent actors/critics (`policy/ddpg_independent.py`) and a parameter‑sharing actor variant (`policy/ddpg_shared.py`). Exploration uses Gaussian action noise; targets are computed with soft‑updated networks.

**MADDPG.** Each agent keeps its own actor; training uses a centralized critic that takes the concatenated observations and actions of all agents (`agents/maddpg.py`). Joint replay is maintained; target actors provide bootstrap targets. Execution remains decentralized.

## Model Architecture

### PPO (agents/ppo.py)

* **Actor**: MLP `obs_dim → 256 → 128 → action_dim`; learnable `log_std` per action.
* **Critic**: MLP `obs_dim → 256 → 128 → 1`.
* **Action bounds**: tanh squashing; stable log‑prob estimation using the unsquashed `u`.

### DDPG / MADDPG (agents/ddpg.py, agents/maddpg.py)

* **Actor**: MLP `obs_dim → 128 → 128 → action_dim` with final `tanh`.
* **Critic**: MLP `(obs_dim + action_dim) → 128 → 128 → 1`.
* **MADDPG critic**: same critic but with concatenated inputs over all agents.

---

## Configuration (excerpt from `configs/ppo.yaml`)

```yaml
learning_rate: 3.0e-4
gamma: 0.99
gae_lambda: 0.95
eps_clip: 0.2
entropy_coef: 0.01
value_loss_coef: 0.5
ppo_epochs: 15
mini_batch_size: 128
learn_every: 5

num_episodes: 5000
max_steps: 500
seed: 42

checkpoint:
  enabled: true
  resume: true
  path: checkpoints/
  freq: 100
```

`configs/default.yaml` varies logging/checkpoint defaults; `configs/ddpg.yaml` defines DDPG/MADDPG hyperparameters (buffer size, batch size, lr, etc.).

---

## Reward shaping & metrics

`utils/metrics.py` contains small helpers you can plug into the environment reward:

* `compute_distance(prev_pos, curr_pos)` — per‑step 2D movement.
* `update_agent_effort(efforts, agent_id, distance)` — accumulate per‑agent effort.
* `compute_shaped_reward(prev_pos, curr_pos, stationary_penalty=..., moving_scale=...)` — example shaping with stationary penalty and (optional) forward bonus.

These are logged via Weights & Biases when enabled.

---

## Results (template)

Fill this table after running your experiments. “Cooperation rate” can be computed using a simple metric from `utils/metrics.py` (e.g., success without dropping the object / joint progress).

| Algorithm                | Average Reward | Cooperation Rate | Notes |
| ------------------------ | -------------: | ---------------: | ----- |
| IDDPG (independent)      |                |                  |       |
| DDPG (parameter sharing) |                |                  |       |
| MADDPG                   |                |                  |       |
| IPPO (parameter sharing) |                |                  |       |

Add plots from W\&B or logs in a `results/` folder and link them here.

---

## Known Issues / Limitations

* `evaluate.py` currently references a legacy DQN path; for IPPO evaluation use `generate_video.py` or adapt `utils/eval_loop.py`.
* Code targets a single machine and specific library versions from the provided conda envs.

<!-- ---

## TODO

* Refactor model loading for shared checkpoints (`utils/load_model.py`).
* Tune / redesign reward shaping to avoid local optima (`utils/metrics.py`).
* Make evaluation compatible with IPPO (`utils/eval_loop.py`).
* Add evaluation metrics for cooperation and per‑agent effort.

---

## License

MIT License (see `LICENSE`). This repository is for coursework; reuse appropriately. -->
