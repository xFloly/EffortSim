#!/usr/bin/env python3

import sys, os, uuid, argparse, json, csv
from datetime import datetime
from omegaconf import OmegaConf, DictConfig
import optuna

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from policy.ppo import run  # uses your existing training loop with LAST_EVAL=True

OPTUNA_STARTUP = 10
OPTUNA_PRUNE_START = 8


def build_cfg(trial: optuna.trial.Trial, base_cfg: DictConfig, trial_group: str) -> DictConfig:
    """Create a cfg for one trial by sampling hyperparameters in wide, mindful ranges."""
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    # ---- mindful, wide ranges (log where appropriate) ----
    cfg.learning_rate   = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
    cfg.gamma           = trial.suggest_float("gamma", 0.97, 0.999)
    cfg.gae_lambda      = trial.suggest_float("gae_lambda", 0.85, 1.00)
    cfg.eps_clip        = trial.suggest_float("eps_clip", 0.10, 0.40)
    cfg.entropy_coef    = trial.suggest_float("entropy_coef", 3e-4, 3e-2, log=True)
    cfg.value_loss_coef = trial.suggest_float("value_loss_coef", 0.2, 0.8)
    cfg.ppo_epochs      = trial.suggest_int("ppo_epochs", 8, 20)
    cfg.mini_batch_size = trial.suggest_categorical("mini_batch_size", [64, 128, 256])
    cfg.learn_every     = trial.suggest_categorical("learn_every", [2, 5, 10])
    cfg.max_steps       = trial.suggest_categorical("max_steps", [500, 700])

    # ---- fixed, fair budget inside a trial ----
    # (we keep num_episodes & eval params constant across seeds for comparability)
    cfg.num_episodes  = base_cfg.num_episodes  
    cfg.eval_episodes = base_cfg.eval_episodes 
    cfg.eval_max_steps = 1000

    # ---- logging (project suffix; group by trial) ----
    base_proj = getattr(cfg, "project", "effortsim")
    cfg.project = f"{base_proj}-optuna"    # emphasize this is sensitivity study
    cfg.group = trial_group

    # ---- checkpoints (isolate per trial; resume off for clean sampling) ----
    cfg.checkpoint.base_path = f"{cfg.checkpoint.path}/optuna/{trial.study.study_name}/trial_{trial.number}"
    cfg.checkpoint.path = None 
    cfg.checkpoint.enabled = True
    cfg.checkpoint.resume = False

    return cfg


def run_one_seed(cfg: DictConfig, seed: int, trial_group: str) -> float:
    """Run training/eval once with a specific base seed, return scalar score."""
    c = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    c.seed = int(seed)

    # unique wandb ID per seed-run
    c.wandb_id = f"t{trial_group}-s{seed}"

    # Consistent checkpoint naming
    c.checkpoint.path = os.path.join(cfg.checkpoint.base_path, f"seed_{seed}")

    score = run(c, LAST_EVAL=True)
    return float(score)


def objective(trial: optuna.trial.Trial, base_cfg: DictConfig, results_writer) -> float:
    """Evaluate one hyperparameter sample across multiple seeds; return mean score."""
    # W&B group name so seed-runs overlay for this trial
    trial_group = f"trial_{trial.study.study_name}_{trial.number}"
    cfg = build_cfg(trial, base_cfg, trial_group)

    # reproducible seed list per trial (distinct, portable)
    seed_list = [240,13,430,2137,900,27,33,8]
    cfg.seed_list = seed_list

    scores = []
    for i, s in enumerate(seed_list, 1):
        score_s = run_one_seed(cfg, s, trial_group)
        scores.append(score_s)

        # write tidy record for later violin/sensitivity plots
        results_writer.writerow({
            "study": trial.study.study_name,
            "trial": trial.number,
            "seed": s,
            "score": score_s,
            # include hyperparams for faceting
            "learning_rate": cfg.learning_rate,
            "gamma": cfg.gamma,
            "gae_lambda": cfg.gae_lambda,
            "eps_clip": cfg.eps_clip,
            "entropy_coef": cfg.entropy_coef,
            "value_loss_coef": cfg.value_loss_coef,
            "ppo_epochs": cfg.ppo_epochs,
            "mini_batch_size": cfg.mini_batch_size,
            "learn_every": cfg.learn_every,
            "max_steps": cfg.max_steps,
            "num_episodes": cfg.num_episodes,
            "eval_episodes": cfg.eval_episodes,
        })

        # report intermediate mean to Optuna (enables pruning)
        trial.report(float(sum(scores) / len(scores)), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # mean across seeds (we do NOT use best-per-seed)
    return float(sum(scores) / len(scores))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="configs/ppo_optuna_sweep.yaml")
    parser.add_argument("--study", type=str, default="ppo_sensitivity")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seeds-per-trial", type=int, default=8, help="Number of seeds per trial (≥5 recommended).")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.base)

    # sampler/pruner—TPE samples the space (we're not grid-searching)
    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=OPTUNA_STARTUP, seed=2025)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=OPTUNA_PRUNE_START, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=args.study,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    # results sink for later violin/sensitivity plots
    os.makedirs("results", exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = os.path.join("results", f"{args.study}_{stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "study", "trial", "seed", "score",
            "learning_rate", "gamma", "gae_lambda", "eps_clip",
            "entropy_coef", "value_loss_coef",
            "ppo_epochs", "mini_batch_size", "learn_every",
            "max_steps", "num_episodes", "eval_episodes"
        ])
        writer.writeheader()

        # optimize (limited #trials → compute aware; we are sampling, not enumerating)
        study.optimize(lambda t: objective(t, base_cfg, writer),
                       n_trials=args.trials, n_jobs=args.n_jobs, gc_after_trial=True)

    # print a quick summary; DO NOT treat as final “best” claim
    print("\nSensitivity study finished.")
    print("Top-trial mean (across seeds):", study.best_value)
    print("Params (for follow-up sensitivity focus):")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    # Save the top-trial params snapshot (for further sensitivity probing—not for headline)
    os.makedirs("best", exist_ok=True)
    best_params_path = os.path.join("best", f"{args.study}_best_params.json")
    with open(best_params_path, "w") as jf:
        json.dump(study.best_trial.params, jf, indent=2)
    print(f"\nSaved top-trial params to {best_params_path}")
    print(f"Tidy per-seed results saved to {csv_path} (use for 1D curves + violin plots).")


if __name__ == "__main__":
    main()
