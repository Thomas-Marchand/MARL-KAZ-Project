import numpy as np

def no_shaping(obs, agent, rewards, infos):
    return 0.0

def proximity_shaping(obs, agent, rewards, infos, reward_scale=0.05):
    """ Reward knights for being closer to the nearest zombie """
    if "knight" not in agent:
        return 0.0
    row_zombie_start, row_zombie_end = 18, 28
    zombie_dists = obs[agent][row_zombie_start:row_zombie_end, 0]
    valid_dists = zombie_dists[zombie_dists > 0]
    if len(valid_dists) == 0:
        return 0.0
    closest_dist = np.min(valid_dists)
    return reward_scale * (1.0 - closest_dist)

def bottom_safety_shaping(obs, agent, rewards, infos, bottom_threshold=0.8, reward_scale=0.1):
    """ Reward agents for no zombies near the bottom, penalize otherwise """
    row_zombie_start, row_zombie_end = 18, 28
    agent_y = obs[agent][0, 2]
    zombie_distances = obs[agent][row_zombie_start:row_zombie_end, 0]
    zombie_relative_ys = obs[agent][row_zombie_start:row_zombie_end, 2]
    valid_zombies = zombie_distances > 0
    if not np.any(valid_zombies):
        return 1.0
    
    max_relative_y = np.max(zombie_relative_ys[valid_zombies])
    max_y = agent_y + max_relative_y
    
    # Smoothing parameters
    width = 0.1
    upper = bottom_threshold - width
    lower = bottom_threshold + width
    
    # Compute reward directly
    if max_y <= upper:
        return 1.0
    elif max_y >= lower:
        return -1.0
    else:
        return 1.0 - 2.0 * (max_y - upper) / (2.0 * width)
