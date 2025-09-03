import os
import uuid
import argparse
from omegaconf import OmegaConf
import optuna

from utils.training_loop_ppo import run  # uses your existing training loop

def build_cfg(trial, base_cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))

    # --- search space (feel free to tweak) ---
    cfg.learning_rate   = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    cfg.gamma           = trial.suggest_float("gamma", 0.97, 0.999)
    cfg.gae_lambda      = trial.suggest_float("gae_lambda", 0.90, 0.99)
    cfg.eps_clip        = trial.suggest_float("eps_clip", 0.1, 0.35)
    cfg.entropy_coef    = trial.suggest_float("entropy_coef", 0.003, 0.03, log=True)
    cfg.value_loss_coef = trial.suggest_float("value_loss_coef", 0.2, 0.8)
    cfg.ppo_epochs      = trial.suggest_int("ppo_epochs", 8, 20)
    cfg.mini_batch_size = trial.suggest_categorical("mini_batch_size", [64, 128, 256])
    cfg.learn_every     = trial.suggest_categorical("learn_every", [2, 5, 10])
    cfg.max_steps       = trial.suggest_categorical("max_steps", [300, 500])

    # --- speed up trials (shorter training) ---
    cfg.num_episodes = min(getattr(cfg, "num_episodes", 10000), 1000)  # e.g., 1000 for HPO

    # --- logging hygiene ---
    cfg.project = f"{getattr(cfg, 'project', 'effortsim')}-optuna"
    wid = f"trial-{trial.number}-{uuid.uuid4().hex[:8]}"
    cfg.wandb_id = wid

    # unique checkpoint dir per trial
    cp_base = getattr(cfg, "checkpoint", None)
    if cp_base is None:
        cfg.checkpoint = {"enabled": True, "resume": False, "path": f"checkpoints/ppo/{wid}", "freq": 100, "mode": "latest"}
    else:
        cfg.checkpoint.enabled = True
        cfg.checkpoint.resume = False
        cfg.checkpoint.path = f"checkpoints/ppo/{wid}"

    return cfg

def objective(trial, base_cfg):
    cfg = build_cfg(trial, base_cfg)
    # You could also do: os.environ["WANDB_MODE"]="offline" to avoid syncing
    score = run(cfg)  # returns best avg100
    # Optuna maximizes by default if we set direction="maximize"
    return score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=str, default="configs/ppo.yaml")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna_study.db")
    parser.add_argument("--study", type=str, default="ppo_optuna")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=1, help="parallel workers")
    args = parser.parse_args()

    base_cfg = OmegaConf.load(args.base)

    sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=10, seed=2025)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=0)

    study = optuna.create_study(
        storage=args.storage,
        study_name=args.study,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(lambda t: objective(t, base_cfg), n_trials=args.trials, n_jobs=args.n_jobs, gc_after_trial=True)

    print("\n== Best trial ==")
    print("value:", study.best_value)
    for k, v in study.best_trial.params.items():
        print(f"{k}: {v}")

    # save best config to YAML for full training
    best_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    for k, v in study.best_trial.params.items():
        OmegaConf.update(best_cfg, k, v, merge=False)

    # restore full training length for the final run
    best_cfg.num_episodes = OmegaConf.load(args.base).get("num_episodes", 10000)

    os.makedirs("best", exist_ok=True)
    out_path = "best/ppo_optuna_best.yaml"
    OmegaConf.save(best_cfg, out_path)
    print(f"\nSaved best config to {out_path}")

if __name__ == "__main__":
    main()
