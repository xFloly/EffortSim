import os
import torch


def load_checkpoints(agents, agent_ids, cfg):
    path = cfg.checkpoint.path
    mode = getattr(cfg.checkpoint, "mode", "latest")

    for aid in agent_ids:
        ckpt_path = None
        if mode == "latest":
            files = [f for f in os.listdir(path) if aid in f and f.endswith(".pt")]
            if files:
                files.sort()
                ckpt_path = os.path.join(path, files[-1])
        elif mode == "individual":
            ckpt_name = cfg.checkpoint.individual_names.get(aid, None)
            if ckpt_name:
                ckpt_path = os.path.join(path, ckpt_name)

        if ckpt_path and os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            agents[aid].actor.load_state_dict(checkpoint['actor_state_dict'])
            agents[aid].actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
            agents[aid].critic.load_state_dict(checkpoint['critic_state_dict'])
            agents[aid].critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            agents[aid].actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            agents[aid].critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
            agents[aid].steps_done = checkpoint.get('steps_done', 0)
            print(f"[Checkpoint] Loaded for {aid} from {ckpt_path}")
        else:
            print(f"[Checkpoint] No checkpoint found for {aid} at {ckpt_path}")
