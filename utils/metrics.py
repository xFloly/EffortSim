import numpy as np

def compute_distance(prev_pos, curr_pos):
    return np.linalg.norm(np.array(curr_pos) - np.array(prev_pos))


def update_agent_effort(effort_dict, agent_id, distance):
    if agent_id not in effort_dict:
        effort_dict[agent_id] = 0.0
    effort_dict[agent_id] += distance
    return effort_dict


def compute_lazy_score(reward, effort):
    """
    Prosty wskaźnik lenistwa: ile agent zyskał względem tego ile pracował.
    Wyższy = bardziej 'leniwy'
    """
    if effort == 0:
        return float("inf") if reward > 0 else 0.0
    return reward / effort


def compute_reward_effort_ratio(agent_rewards, agent_efforts):
    """Zwraca słownik: agent_id -> stosunek nagrody do wysiłku"""
    return {
        aid: compute_lazy_score(agent_rewards.get(aid, 0.0), agent_efforts.get(aid, 0.0))
        for aid in set(agent_rewards) | set(agent_efforts)
    }


def compute_effort_variance(agent_efforts):
    """Wariancja wysiłku agentów (proxy dla nierówności)"""
    values = list(agent_efforts.values())
    return np.var(values) if values else 0.0


def log_metrics_to_wandb(efforts, rewards, step=None):
    """Logging agent metrics to wandb"""
    import wandb

    ratios = compute_reward_effort_ratio(rewards, efforts)
    for aid in efforts:
        wandb.log({
            f"{aid}/effort": efforts[aid],
            f"{aid}/reward": rewards.get(aid, 0.0),
            f"{aid}/lazy_score": ratios[aid]
        }, step=step)

    wandb.log({
        "effort_variance": compute_effort_variance(efforts)
    }, step=step)
