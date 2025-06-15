import numpy as np
import wandb

def compute_distance(prev_pos, curr_pos):
    """Euclidean distance between two positions."""
    return np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))

def update_agent_effort(effort_dict, agent_id, distance):
    """Accumulate movement distance per agent."""
    effort_dict[agent_id] = effort_dict.get(agent_id, 0.0) + distance
    return effort_dict

def log_metrics_to_wandb(efforts, rewards, step=None):
    """Log raw reward and effort to wandb, per agent."""
    for aid in efforts:
        wandb.log({
            f"{aid}/effort": efforts[aid],
            f"{aid}/reward": rewards.get(aid, 0.0),
        }, step=step)
