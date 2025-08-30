import os
import glob
import torch


def save_maddpg_checkpoints(maddpg, agent_ids, cfg, episode, total_steps, final=False):
    """
    Save checkpoints for MADDPG:
      - one file per actor
      - one file for the centralized critic
    """
    if not cfg.checkpoint.enabled:
        return

    os.makedirs(cfg.checkpoint.path, exist_ok=True)
    suffix = "final" if final else f"ep{episode+1}"

    # save each actor
    for aid in agent_ids:
        ckpt_actor = {
            "actor_state_dict": maddpg.actors[aid].state_dict(),
            "actor_target_state_dict": maddpg.actors_target[aid].state_dict(),
            "actor_optimizer_state_dict": maddpg.actor_optimizers[aid].state_dict(),
            "steps_done": total_steps,
            "episode": episode + 1,
        }
        torch.save(
            ckpt_actor,
            os.path.join(cfg.checkpoint.path, f"{aid}_maddpg_actor_{suffix}.pt"),
        )

    # save centralized critic
    ckpt_critic = {
        "critic_state_dict": maddpg.critic.state_dict(),
        "critic_target_state_dict": maddpg.critic_target.state_dict(),
        "critic_optimizer_state_dict": maddpg.critic_optimizer.state_dict(),
        "steps_done": total_steps,
        "episode": episode + 1,
    }
    torch.save(
        ckpt_critic,
        os.path.join(cfg.checkpoint.path, f"central_critic_maddpg_{suffix}.pt"),
    )

    print(f"[Checkpoint] Saved MADDPG ({'final' if final else f'episode {episode+1}'})")


def load_maddpg_checkpoints(maddpg, agent_ids, cfg):
    """
    Load MADDPG checkpoints for all actors + centralized critic.
    Returns (total_steps, start_episode) so training can resume.
    """
    path = cfg.checkpoint.path
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
