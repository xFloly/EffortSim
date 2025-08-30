import numpy as np
import wandb

def compute_distance(prev_pos, curr_pos):
    """Compute the Euclidean distance between two 2D positions."""
    return np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))

def update_agent_effort(effort_dict, agent_id, distance):
    """
    Accumulate total movement effort (distance traveled) for a specific agent.

    Parameters:
        effort_dict (dict): A dictionary tracking movement per agent.
        agent_id (str): The ID of the agent to update.
        distance (float): The distance moved in the current step.

    Returns:
        dict: Updated effort dictionary.
    """
    effort_dict[agent_id] = effort_dict.get(agent_id, 0.0) + distance
    return effort_dict

def log_metrics_to_wandb(efforts, rewards, step=None):
    """
    Log movement effort and total reward per agent to Weights & Biases.

    Parameters:
        efforts (dict): Mapping of agent ID to cumulative effort.
        rewards (dict): Mapping of agent ID to cumulative reward.
        step (int, optional): Optional global training step or episode number.
    """
    for aid in efforts:
        wandb.log({
            f"{aid}/effort": efforts[aid],
            f"{aid}/reward": rewards.get(aid, 0.0),
        }, step=step)

def penalty(curr_pos, prev_pos, moving_scale=5, stationary_penalty=-10.0, fall_penalty_scale=5):
    """
    Compute a custom movement penalty/reward based on positional change.

    Behavior:
        - Penalizes agents for standing still in the X direction.
        - (Optional/test) Rewards agents for moving forward in X.
        - (Optional/test) Penalizes agents for falling down in Z (lower Y-coordinate).

    Parameters:
        curr_pos (tuple): Current (x, y) position of the agent.
        prev_pos (tuple): Previous (x, y) position of the agent.
        moving_scale (float): Scaling factor for rewarding movement (currently unused).
        stationary_penalty (float): Penalty for standing still in X.
        fall_penalty_scale (float): Scaling factor for penalizing downward movement in Z/Y (currently unused).

    Returns:
        float: Calculated reward/penalty.
    """
    reward = 0.0

    # Forward movement in the X direction
    dx = curr_pos[0] - prev_pos[0]

    # Penalize if agent is nearly stationary
    if dx <= 0.01:
        reward += stationary_penalty

    # Optional: reward forward movement (commented out for testing)
    # reward += dx * moving_scale

    # Optional: penalize falling (downward movement in Z/Y)
    # dz = curr_pos[1] - prev_pos[1]  # Higher Z is better
    # if dz < 0:
    #     reward += dz * fall_penalty_scale  # Negative dz results in a penalty

    return reward
