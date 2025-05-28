import os
import torch
import glob

def load_checkpoints(agents, agent_ids, cfg):
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
        ckpt_name = cfg.checkpoint.shared_name
        ckpt_path = os.path.join(path, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"shared checkpoint {ckpt_name} not found")
            return
        for aid in agent_ids:
            _load(agents[aid], ckpt_path, cfg)
            print(f"walker {aid} initialized from shared checkpoint {ckpt_path}")

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
    files = sorted(
        glob.glob(os.path.join(path, f"{agent_id}_checkpoint*.pt")),
        key=os.path.getmtime,
        reverse=True
    )
    return files[0] if files else None

def _load(agent, ckpt_path, cfg):
    checkpoint = torch.load(ckpt_path, weights_only=True)
    agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
    agent.target_net.load_state_dict(checkpoint['target_state_dict'])
    agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.epsilon = checkpoint.get('epsilon', cfg.eps_start)
    agent.steps_done = checkpoint.get('steps_done', 0)
