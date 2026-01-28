from colorama import init, Style, Fore
init(autoreset=True)
import numpy as np
from pettingzoo.butterfly import knights_archers_zombies_v10


def fmt(row): # Format row of numbers
    out = []
    for val in row:
        s_val = f"{val:>7.3f}"
        if abs(val) < 1e-6: # 0
            out.append(f"{Style.DIM}{s_val}{Style.RESET_ALL}")
        else:
            out.append(s_val)
    return " ".join(out)

# --- Configuration ---
num_archers = 2
num_knights = 2 
max_zombies = 10
max_arrows = 10

env = knights_archers_zombies_v10.parallel_env(
    num_archers=num_archers, num_knights=num_knights, max_zombies=max_zombies, max_arrows=max_arrows,
    spawn_rate=20, # default 20
    killable_knights=True,
    killable_archers=True,
    pad_observation=True,
    line_death=False,
    max_cycles=900,
    vector_state=True,
    use_typemasks=False,
    sequence_space=False,
    render_mode="human"
)

observations, infos = env.reset(seed=42)
cum_rewards = {agent: 0 for agent in env.possible_agents}

# Offsets based on environment logic
offset_swords = 2 + num_archers + num_knights # pos+unit + 2 + 2 = 6
offset_arrows = offset_swords + num_knights # 8
offset_zombies = offset_arrows + max_arrows

all_agents = sorted(env.possible_agents) # to maintain column even when some die

step_count = 0

while env.agents:
    step_count += 1
    # Step Environment
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    
    print(f"\n{'='*40} STEP {step_count} {'='*40}")

    header = f"{' ':>12} |"
    sub_header = f"{'Entity':>12} |"
    
    for agent in all_agents:
        status_color = Fore.GREEN if agent in env.agents else Fore.RED
        header += f"{status_color}{agent:^25}{Style.RESET_ALL}|"
        sub_header += f" {'D':>7} {'X':>7} {'Y':>7} |"
    
    sub_header += f" {'U':>7} {'V':>7} |"
    
    print(header)
    print(sub_header)
    print("-" * len(sub_header))

    row_pos = f"{'Self Pos':>12} |"
    row_vec = f"{'Self Vec':>12} |"
    
    for agent in all_agents:
        if agent in observations:
            obs = observations[agent]
            p_str = fmt(obs[0, 1:3])
            v_str = fmt(obs[0, 3:5])
            row_pos += f" {Style.DIM}{'-':>7}{Style.RESET_ALL} {p_str} |"
            row_vec += f" {Style.DIM}{'-':>7}{Style.RESET_ALL} {v_str} |"
        else:
            row_pos += f"{Style.DIM}{'DEAD':^24}{Style.RESET_ALL}|"
            row_vec += f"{Style.DIM}{'DEAD':^24}{Style.RESET_ALL}|"
    
    row_pos += f"{Style.DIM}{'-':>8} {'-':>7}{Style.RESET_ALL} |"
    row_vec += f"{Style.DIM}{'-':>8} {'-':>7}{Style.RESET_ALL} |"
            
    print(row_pos)
    print(row_vec)
    print("-" * len(sub_header))

    def print_entity_rows(label_prefix, start_idx, count, obs_offset_idx):
        for i in range(count):
            label = f"{label_prefix}_{i}"
            row_str = f"{label:>12} |"
            u_val = None
            v_val = None
            
            for agent in all_agents:
                if agent in observations:
                    obs = observations[agent]
                    matrix_row_idx = obs_offset_idx + i 
                    row_data = obs[matrix_row_idx]
                    dxy_str = fmt(row_data[:3])
                    row_str += f" {dxy_str} |"
                    if u_val is None:
                        u_val = row_data[3]
                        v_val = row_data[4]
                else:
                    row_str += f"{Style.DIM}{' ':^24}{Style.RESET_ALL}|"
            row_str += f" {u_val:>7.3f} {v_val:>7.3f} |"
            print(row_str)

    print_entity_rows("🏹 archer", 0, num_archers, 1)
    print_entity_rows("🪖 knight", 0, num_knights, 1 + num_archers)
    
    print("-" * len(sub_header))
    
    print_entity_rows("  🗡️  sword", 0, num_knights, offset_swords - 1)
    print_entity_rows("↗  arrow", 0, max_arrows, offset_arrows - 1)
    
    print("-" * len(sub_header))

    print_entity_rows("🧟 zombie", 0, max_zombies, offset_zombies - 1)
    
    for agent, r in rewards.items():
        cum_rewards[agent] += r
        
    rew_str = f"{'Rewards':>13} |"
    for agent in all_agents:
        if agent in cum_rewards:
            rew_str += f"{f'{rewards.get(agent,0):.1f} (Tot:{cum_rewards[agent]:.1f})':^25}|"
        else:
            rew_str += f"{' ':^24}|"
    print(rew_str)

env.close()