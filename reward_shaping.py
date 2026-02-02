import numpy as np

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
        return reward_scale
    
    max_relative_y = np.max(zombie_relative_ys[valid_zombies])
    max_y = agent_y + max_relative_y
    
    # Smoothing parameters
    width = 0.1
    upper = bottom_threshold - width
    lower = bottom_threshold + width
    
    # Compute base reward
    if max_y <= upper:
        return reward_scale
    elif max_y >= lower:
        return -reward_scale
    else:
        base_reward = 1.0 - 2.0 * (max_y - upper) / (2.0 * width)
    
    return reward_scale * base_reward

def position_shaping(obs, agent, rewards, infos, n_archers, n_knights, reward_scale=0.1):
    """ Reward agents for being scattered efficiently (knights high & scattered, archers low & scattered) """
    optimal_knight_y = 0.33
    optimal_archer_y = 0.67
    agent_type = "archer" if "archer" in agent else "knight"
    agent_number = int(agent.split("_")[1])
    agent_x, agent_y = obs[agent][0, 1], obs[agent][0, 2]
    
    if agent_type == "knight":
        optimal_ys = [optimal_knight_y] * n_knights
        optimal_xs = [(1.0 / (n_knights + 1)) * (i + 1) for i in range(n_knights)]
        n_agents = n_knights
    else:
        optimal_ys = [optimal_archer_y] * n_archers
        optimal_xs = [(1.0 / (n_archers + 1)) * (i + 1) for i in range(n_archers)]
        n_agents = n_archers
    
    reward = 0.0
    decay_factor = 2.0 # controls how quickly the reward decays with distance
    for i in range(n_agents):
        dist = np.sqrt((agent_x - optimal_xs[i]) ** 2 + (agent_y - optimal_ys[i]) ** 2)
        exp_reward = reward_scale * np.exp(-dist * decay_factor)
        if i == agent_number:
            reward += exp_reward
        else:
            reward -= exp_reward
    
    # print(f"Agent: {agent}, Type: {agent_type}, Pos: ({agent_x:.2f}, {agent_y:.2f}), Reward: {reward:.4f}")
    # print(f"  Optimal positions: Xs: {optimal_xs}, Ys: {optimal_ys}")
    # print(f"  Distances to optimals: {[float(np.sqrt((agent_x - ox) ** 2 + (agent_y - oy) ** 2)) for ox, oy in zip(optimal_xs, optimal_ys)]}")
    # print(f"  Exp rewards: {[float(reward_scale * np.exp(-np.sqrt((agent_x - ox) ** 2 + (agent_y - oy) ** 2) * decay_factor)) for ox, oy in zip(optimal_xs, optimal_ys)]}")
    # print("-----")
    
    return reward