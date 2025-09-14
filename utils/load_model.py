import warnings
warnings.filterwarnings("ignore", message=".*weights_only=False.*", category=FutureWarning)

import os
import torch
import glob

def load_checkpoints(agents, agent_ids, cfg, path):
    ### Get latest checkpoint file for specific agent ###
    mode = cfg.checkpoint.mode
    loaded_any = False

    if mode == "latest":
        for aid in agent_ids:
            ckpt = _get_latest_ckpt(path, aid)
            if ckpt:
                _load(agents[aid], ckpt, cfg)
                print(f"walker {aid} initialized from checkpoint {ckpt}")
                loaded_any = True
            else:
                print(f"walker {aid} skipped (no latest checkpoint found)")


    elif mode == "shared":
        ckpt_name = cfg.checkpoint.shared_name
        ckpt_path = os.path.join(path, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[warn] shared checkpoint not found: {ckpt_path}")
        else:
            for aid in agent_ids:
                _load(agents[aid], ckpt_path, cfg)
                print(f"walker {aid} initialized from shared checkpoint {ckpt_path}")
                loaded_any = True

    elif mode == "individual":
        names = cfg.checkpoint.individual_names
        for aid in agent_ids:
            ckpt_name = names.get(aid, None)
            if ckpt_name is None:
                print(f"walker {aid} skipped (no checkpoint specified)")
                continue
            ckpt_path = os.path.join(path, ckpt_name)
            if not os.path.exists(ckpt_path):
                print(f"walker {aid} skipped (file not found: {ckpt_path})")
                continue
            _load(agents[aid], ckpt_path, cfg)
            print(f"walker {aid} initialized from checkpoint {ckpt_path}")
            loaded_any = True
    return loaded_any

def _get_latest_ckpt(path, agent_id):
    import re

    ### if final checkpoint exists load it
    final_ckpt = os.path.join(path, f"{agent_id}_checkpoint_final.pt")
    print(f"[info] loading checkpoint {final_ckpt}")
    if os.path.exists(final_ckpt):
        return final_ckpt

    ### Get latest checkpoint file for specific agent ###
    def extract_ep_num(filename):
        match = re.search(r"shared_checkpoint_ep(\d+)\.pt", filename)
        return int(match.group(1)) if match else -1

    files = sorted(
        glob.glob(os.path.join(path, "shared_checkpoint_ep*.pt")),
        key=extract_ep_num,
        reverse=True
    )
    return files[0] if files else None

def _load(agent, ckpt_path, cfg):
    checkpoint = torch.load(ckpt_path, map_location=agent.device)

    # ── nowy format ─────────────────────────────
    if "actor_state_dict" in checkpoint:
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    # ── stary format (wsteczna kompat.) ─────────
    else:
        agent.actor.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if "optimizer_state_dict" in checkpoint:
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    agent.episode    = checkpoint.get("episode", 0)
    agent.steps_done = checkpoint.get("steps_done", 0)



def load_agent(agent_name, model_path, env, device="cpu", cfg=None):

    agent_ids = env.possible_agents
    obs_dim = env.observation_space(agent_ids[0]).shape[0]
    action_dim = env.action_space(agent_ids[0]).shape[0]

    if agent_name == "ppo":
        from agents.ppo import PPOAgent
        agent = PPOAgent("evalPPO", obs_dim, action_dim, device=device, cfg=cfg)
    elif agent_name == "ddpg":
        from agents.ddpg import DDPGAgent
        agent = DDPGAgent("evalDDPG", obs_dim, action_dim, device=device, cfg=cfg)
    elif agent_name == "maddpg":
        from agents.maddpg import MADDPG
        agent = MADDPG(agent_ids, obs_dim, action_dim, device=device, cfg=cfg)
    else:
        raise ValueError(f"Unknown agent: {agent_name}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    if "actor_state_dict" in checkpoint:
        agent.actor.load_state_dict(checkpoint["actor_state_dict"])
        if hasattr(agent, "critic"):
            agent.critic.load_state_dict(checkpoint["critic_state_dict"])
    elif "model_state_dict" in checkpoint: # old format
        agent.actor.load_state_dict(checkpoint["model_state_dict"], strict=False)

    if "optimizer_state_dict" in checkpoint and hasattr(agent, "optimizer"):
        agent.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    agent.episode    = checkpoint.get("episode", 0)
    agent.steps_done = checkpoint.get("steps_done", 0)

    print(f"walker {agent_name} initialized from checkpoint {model_path}")

    agent.actor.eval()
    if hasattr(agent, "critic"):
        agent.critic.eval()

    return agent


def load_maddpg_checkpoints(maddpg, agent_ids, cfg, path):
    """
    Load MADDPG checkpoints for all actors + centralized critic.
    Returns (total_steps, start_episode) so training can resume.
    """
    # path = cfg.checkpoint.path
    loaded_any = False
    total_steps = 0
    start_episode = 0

    # --- load actors ---
    for aid in agent_ids:
        ckpt = _get_latest_file(path, f"{aid}_maddpg_actor")
        if ckpt:
            checkpoint = torch.load(ckpt, map_location=maddpg.device)
            maddpg.actors[aid].load_state_dict(checkpoint["actor_state_dict"])
            maddpg.actors_target[aid].load_state_dict(checkpoint["actor_target_state_dict"])
            maddpg.actor_optimizers[aid].load_state_dict(checkpoint["actor_optimizer_state_dict"])

            total_steps = max(total_steps, checkpoint.get("steps_done", 0))
            start_episode = max(start_episode, checkpoint.get("episode", 0))

            print(f"[Load] Actor {aid} restored from {ckpt}")
            loaded_any = True
        else:
            print(f"[Load] No checkpoint found for actor {aid}")

    # --- load central critic ---
    ckpt = _get_latest_file(path, "central_critic_maddpg")
    if ckpt:
        checkpoint = torch.load(ckpt, map_location=maddpg.device)
        maddpg.critic.load_state_dict(checkpoint["critic_state_dict"])
        maddpg.critic_target.load_state_dict(checkpoint["critic_target_state_dict"])
        maddpg.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])

        total_steps = max(total_steps, checkpoint.get("steps_done", 0))
        start_episode = max(start_episode, checkpoint.get("episode", 0))

        print(f"[Load] Central critic restored from {ckpt}")
        loaded_any = True
    else:
        print("[Load] No checkpoint found for central critic")

    return loaded_any, total_steps, start_episode



def _get_latest_file(path, prefix):
    """
    Find the latest checkpoint for files starting with prefix.
    Prioritizes *_final.pt, otherwise finds max episode number.
    """
    final_ckpt = os.path.join(path, f"{prefix}_final.pt")
    if os.path.exists(final_ckpt):
        return final_ckpt

    files = glob.glob(os.path.join(path, f"{prefix}_ep*.pt"))
    if not files:
        return None

    def extract_ep(filename):
        import re
        match = re.search(r"ep(\d+)\.pt", filename)
        return int(match.group(1)) if match else -1

    files = sorted(files, key=extract_ep, reverse=True)
    return files[0]
