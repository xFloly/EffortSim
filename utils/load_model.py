import os
import torch
import glob

def load_checkpoints(agents, agent_ids, cfg):
    ### Get latest checkpoint file for specific agent ###
    mode = cfg.checkpoint.mode
    path = cfg.checkpoint.path

    if mode == "latest":
        for aid in agent_ids:
            ckpt = _get_latest_ckpt(path, aid)
            if ckpt:
                _load(agents[aid], ckpt, cfg)
                print(f"walker {aid} initialized from checkpoint {ckpt}")
            else:
                print(f"walker {aid} skipped (no latest checkpoint found)")

    elif mode == "shared":
        # TODO
        pass
    
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

def _get_latest_ckpt(path, agent_id):
    import re

    ### if final checkpoint exists load it
    final_ckpt = os.path.join(path, f"{agent_id}_checkpoint_final.pt")
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
    ### Load model weights and training state into an agent ###
    checkpoint = torch.load(ckpt_path, weights_only=True)
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
    agent.critic.load_state_dict(checkpoint['critic_state_dict'])
    agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    agent.steps_done = checkpoint.get('steps_done', 0)
    agent.episode = checkpoint.get('episode',0)
