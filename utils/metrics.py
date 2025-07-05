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

def penalty(curr_pos, prev_pos, moving_scale=5, stationary_penalty=-3., fall_penalty_scale=5):
    """
    movement penalty/nagroda: nagroda za ruch do przodu
    + kara za stanie w miejscu
    + kara za spadanie (zmniejszanie Z)
    """
    reward = 0.0
    # Ruch w poziomie (X)
    dx = curr_pos[0] - prev_pos[0]
    # reward = dx * moving_scale

    # Kara za stanie w miejscu
    if dx <= 0.01:
        reward += stationary_penalty
    # else:
    #     reward += moving_scale

    # Kara za spadanie (Z)
    # dz = curr_pos[1] - prev_pos[1]  # wyżej = pozytywne
    # if dz < 0:
    #     reward += dz * fall_penalty_scale  # dz < 0 → minus → kara

    return reward
