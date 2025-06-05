import numpy as np

def compute_distance(pos1, pos2):
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def update_agent_effort(efforts, aid, dist):
    efforts[aid] += dist

def log_metrics_to_wandb(agent_efforts, agent_rewards, step):
    # Optionally log metrics here if you use wandb
    # Just a placeholder, you can expand this:
    print(f"[Step {step}] Efforts: {agent_efforts}, Rewards: {agent_rewards}")
