import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import supersuit as ss
import multiprocessing
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.utils import set_random_seed
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.wrappers import BaseParallelWrapper
from reward_shaping import proximity_shaping, bottom_safety_shaping, position_shaping

# === CONFIG ===

TOTAL_TIMESTEPS = 20000000

N_RUNS_PER_SCENARIO = 4
SPAWN_RATE = 5 # every X steps
NUM_ARCHERS, NUM_KNIGHTS = 2, 2
MAX_ZOMBIES = 10
MAX_ARROWS = 10
PARALLEL_ENVS = 8
N_STEPS_PPO = 2048

N_CPUS = multiprocessing.cpu_count()
AGENTS_PER_STEP = (NUM_ARCHERS + NUM_KNIGHTS) * PARALLEL_ENVS
STEP_SIZE = N_STEPS_PPO * AGENTS_PER_STEP
REAL_TOTAL_TIMESTEPS = ((TOTAL_TIMESTEPS + STEP_SIZE - 1) // STEP_SIZE) * STEP_SIZE

# COLORS for scenarios
COLORS = {
    "Base_Reward": "#ff5353",
    "Proximity_Shaping": "#2a9dff",
    "Bottom_Safety_Shaping": "#28a745",
    "Mixed_Shaping_(prox+bottom)": "#ffa500",
    "Mixed_Shaping_(prox+bottom+pos)": "#800080"
}

# SCENARIOS: (name, use_shaping, shapings_list)
# Each shaping in shapings_list: {"func": func, "kwargs": {}, "start_pct": 0.0, "end_pct": 1.0}
SCENARIOS = [
    ("Base Reward", False, []),
    # ("Proximity Shaping", True, [
    #     {"func": proximity_shaping, "kwargs": {"reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 1.0}
    # ]),
    # ("Bottom Safety Shaping", True, [
    #     {"func": bottom_safety_shaping, "kwargs": {"bottom_threshold": 0.8, "reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 1.0}
    # ]),
    ("Position Shaping", True, [
        {"func": position_shaping, "kwargs": {"n_archers": NUM_ARCHERS, "n_knights": NUM_KNIGHTS, "reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 1.0}
    ]),
    ("Mixed Shaping_(prox+bottom)", True, [
        {"func": proximity_shaping, "kwargs": {"reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 0.5},
        {"func": bottom_safety_shaping, "kwargs": {"bottom_threshold": 0.8, "reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 0.3}
    ]),
    ("Mixed Shaping_(prox+bottom+pos)", True, [
        {"func": proximity_shaping, "kwargs": {"reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 0.5},
        {"func": bottom_safety_shaping, "kwargs": {"bottom_threshold": 0.8, "reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 0.3},
        {"func": position_shaping, "kwargs": {"n_archers": NUM_ARCHERS, "n_knights": NUM_KNIGHTS, "reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 0.5},
    ])
]

OUTPUT_DIR = f"./RShaping_{PARALLEL_ENVS}x{N_RUNS_PER_SCENARIO}x{REAL_TOTAL_TIMESTEPS//1000}K/"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

AGENT_NAMES = [f"archer_{i}" for i in range(NUM_ARCHERS)] + [f"knight_{i}" for i in range(NUM_KNIGHTS)]

print(f"Output Directory: {OUTPUT_DIR}")
print(f"Total Timesteps per run: {REAL_TOTAL_TIMESTEPS} ~ {REAL_TOTAL_TIMESTEPS//1000}K")

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

# For smoothing the plot lines
SMOOTH_WINDOW = 500 if REAL_TOTAL_TIMESTEPS >= 5_000_000 else 100


# ==============
#    TRAINING
# ==============

# Shaping
class RewardShapingWrapper(BaseParallelWrapper):
    def __init__(self, env, shapings, total_timesteps):
        super().__init__(env)
        self.shapings = shapings
        self.total_timesteps = total_timesteps
        self.step_counter = 0
    
    def step(self, actions):
        self.step_counter += 1
        current_pct = self.step_counter / self.total_timesteps if self.total_timesteps else 0.0
        
        obs, rewards, terminations, truncations, infos = super().step(actions)
        
        for agent in self.agents:
            shaped_reward = 0.0
            for shaping in self.shapings:
                if shaping['start_pct'] <= current_pct <= shaping['end_pct']:
                    shaped_reward += shaping['func'](obs, agent, rewards, infos, **shaping['kwargs'])
            
            original_reward = float(rewards[agent])
            rewards[agent] = original_reward + shaped_reward
            
            # Store shaping info in the info dict for the callback to see
            if agent in infos:
                infos[agent]["shaping_reward"] = shaped_reward
                infos[agent]["original_reward"] = original_reward
        
        # dead
        for agent in set(self.possible_agents) - set(self.agents):
            if agent in infos:
                infos[agent]["original_reward"] = float(rewards.get(agent, 0))
                infos[agent]["shaping_reward"] = 0.0
        
        return obs, rewards, terminations, truncations, infos


# CUSTOM CALLBACK
class AgentBreakdownCallback(BaseCallback):
    def __init__(self, run_idx, use_shaping=False, verbose=0):
        super().__init__(verbose)
        self.run_idx = run_idx
        self.use_shaping = use_shaping
        self.history = {name: [] for name in AGENT_NAMES}
        self.history['team'] = []
        self.history['length'] = []
        self.history['timestamps'] = []
        self.history['shaping_sum'] = []
        self.history['true_team'] = []
        self.current_shaping = {name: 0.0 for name in AGENT_NAMES}
        self.current_original = {i: {name: 0.0 for name in AGENT_NAMES} for i in range(PARALLEL_ENVS)}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos: return True

        # Accumulate shaping rewards per agent
        for i, info in enumerate(infos):
            env_idx = i // 4
            agent_idx = i % 4
            agent_name = AGENT_NAMES[agent_idx]
            shaped = info.get("shaping_reward", 0)
            self.current_shaping[agent_name] += shaped
            original = info.get("original_reward", 0)
            self.current_original[env_idx][agent_name] += original

        # Parallel envs -> 4 agents at a time (2 archers, 2 knights) (or different if the parameters have been changed, todo clean)
        for env_idx in range(0, len(infos), 4):
            env_num = env_idx // 4
            if "episode" in infos[env_idx]:
                current_ts = self.num_timesteps
                ep_stat = infos[env_idx]["episode"]
                
                self.history['timestamps'].append(current_ts)
                self.history['length'].append(ep_stat["l"])

                team_reward = 0
                shaping_total = 0
                true_team = 0
                true_rewards = []
                for agent_idx, agent_name in enumerate(AGENT_NAMES):
                    actual_idx = env_idx + agent_idx
                    # Base/Total Reward
                    r = infos[actual_idx]["episode"]["r"] if "episode" in infos[actual_idx] else 0
                    if self.use_shaping:
                        true_r = self.current_original[env_num][agent_name]
                    else:
                        true_r = r
                    self.history[agent_name].append(true_r)
                    team_reward += r
                    # Track shaped component from accumulated
                    shaping_total += self.current_shaping[agent_name]
                    true_team += true_r
                    true_rewards.append(true_r)
                
                self.history['team'].append(true_team)
                self.history['shaping_sum'].append(shaping_total)
                self.history['true_team'].append(true_team)

                if env_idx == 0:
                    ep_count = len(self.history['team'])
                    true_str = ", ".join([f"{r:>4.0f}" for r in true_rewards])
                    shaping_str = ", ".join([f"{self.current_shaping[name]:.3f}" for name in AGENT_NAMES])
                    pct = (current_ts/REAL_TOTAL_TIMESTEPS)*100
                    print(f"Run {self.run_idx:<2}: {pct:5.1f}% | True Rew: [{true_str}] | Tot: {true_team:4.0f} | Shaping: [{shaping_str}]")
                
                # reset for next episode
                self.current_shaping = {name: 0.0 for name in AGENT_NAMES}
                self.current_original[env_num] = {name: 0.0 for name in AGENT_NAMES}
            
        return True

    def get_data(self):
        return {k: np.array(v) for k, v in self.history.items()}


def make_env(use_shaping, render_mode=None, num_envs=8, shapings=None, total_timesteps=None):
    env = knights_archers_zombies_v10.parallel_env(
        spawn_rate=SPAWN_RATE, num_archers=NUM_ARCHERS, num_knights=NUM_KNIGHTS,
        max_zombies=MAX_ZOMBIES, max_arrows=MAX_ARROWS,
        vector_state=True, use_typemasks=False, render_mode=render_mode
    )
    if use_shaping and shapings:
        env = RewardShapingWrapper(env, shapings, total_timesteps)
    env = ss.black_death_v3(env)
    env = ss.flatten_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    if num_envs > 1:
        env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=0, base_class='stable_baselines3')
    env = VecMonitor(env)
    return env





def train_single_run(args):
    name, use_shaping, shapings, run_idx = args
    run_name = f"{name.replace(' ', '_')}_run{run_idx}"
    set_random_seed(run_idx * 42)
    
    print(f"STARTING: {name} | RUN: {run_idx}")
    start_time = time.time()
    env = make_env(use_shaping=use_shaping, num_envs=PARALLEL_ENVS, shapings=shapings, total_timesteps=REAL_TOTAL_TIMESTEPS)
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, batch_size=2048, n_steps=N_STEPS_PPO)
    callback = AgentBreakdownCallback(run_idx, use_shaping=use_shaping)
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    runtime = time.time() - start_time
    data = callback.get_data()
    data['runtime'] = runtime

    np.savez(os.path.join(LOG_DIR, f"stats_{run_name}.npz"), **data)
    model.save(os.path.join(MODELS_DIR, f"ppo_{run_name}"))
    env.close()
    
    print(f"FINISHED: {name} | RUN: {run_idx} | Runtime: {runtime:.2f}s")
    return True

def run_training():
    scenarios = SCENARIOS
    tasks = [(name, use_shaping, shapings, i) for name, use_shaping, shapings in scenarios for i in range(1, N_RUNS_PER_SCENARIO + 1)]

    print(f"Starting Multi-processing training for {len(tasks)} runs, {REAL_TOTAL_TIMESTEPS} timesteps each...")
    with multiprocessing.Pool(processes=N_CPUS) as pool:
        pool.map(train_single_run, tasks)




# ==============
#    PLOTTING
# ==============

def smooth_data(values, window=20):
    if len(values) < window:
        return values
    smoothed = np.zeros_like(values, dtype=float)
    half_window = window // 2
    for i in range(len(values)):
        start = max(0, i - half_window)
        end = min(len(values), i + half_window + 1)
        smoothed[i] = np.mean(values[start:end])
    return smoothed

def run_plotting(plot_individual=False):
    print(f"\nGenerating plots from logs, plot_individual={plot_individual}...")
    scenario_names = [name.replace(' ', '_') for name, _, _ in SCENARIOS]
    scenarios = scenario_names
    colors = {name: COLORS.get(name, "#000000") for name in scenario_names}

    all_raw_data = {s: [] for s in scenarios}
    runtimes = {s: [] for s in scenarios}
    max_iter = 0
    min_iter = float('inf')

    for name in scenarios:
        files = glob.glob(os.path.join(LOG_DIR, f"stats_{name}_run*.npz"))
        for f in files:
            d = np.load(f)
            iters = d['timestamps'] // AGENTS_PER_STEP
            if len(iters) > 0:
                min_iter = min(min_iter, iters[0])
                max_iter = max(max_iter, iters[-1])
            all_raw_data[name].append(d)
            if 'runtime' in d:
                runtimes[name].append(d['runtime'])

    for name in scenarios:
        num_runs = len(all_raw_data[name])
        total_episodes = sum(len(d['team']) for d in all_raw_data[name]) if all_raw_data[name] else 0
        avg_runtime = np.mean(runtimes[name]) if runtimes[name] else 0
        print(f"Scenario '{name}': {num_runs} runs, {total_episodes} total episodes, avg runtime {avg_runtime:.1f}s.")

    if max_iter == 0:
        print("No data found to plot."); return

    # Stats
    scenario_stats = {}
    for name in scenarios:
        num_runs = len(all_raw_data[name])
        total_episodes = sum(len(d['team']) for d in all_raw_data[name]) if all_raw_data[name] else 0
        avg_runtime = np.mean(runtimes[name]) if runtimes[name] else 0
        scenario_stats[name] = {'runs': num_runs, 'episodes': total_episodes, 'avg_runtime': avg_runtime}

    common_x = np.linspace(min_iter, max_iter, 1000)
    metrics = ['team', 'archers', 'knights', 'length']
    fig, axs = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    metric_positions = {'team': (0, 0), 'archers': (0, 1), 'knights': (1, 1), 'length': (1, 0)}

    individual_alpha = 0.2

    # Suptitle w stats
    line1 = f"Training Analysis: {N_RUNS_PER_SCENARIO} runs each, Smoothing window={SMOOTH_WINDOW}, Mean with 25-75% percentiles"
    if plot_individual:
        line1 += ", Individual runs shown"
    line2 = ", ".join([f"{name}: {scenario_stats[name]['episodes']} episodes, avg {scenario_stats[name]['avg_runtime']:.1f}s" for name in scenarios])
    fig.suptitle(f"{line1}\n{line2}", fontsize=10, fontweight='bold')

    mean_ys = {metric: [] for metric in metrics}
    percentile_ys = {metric: {'p25': [], 'p75': []} for metric in metrics}

    for metric in metrics:
        ax = axs[metric_positions[metric]]
        
        # Title
        base_title = metric.capitalize()
        ax.set_title(base_title, fontsize=14, fontweight='bold')
        
        for name in scenarios:
            if metric in ['team', 'length']:
                interp_runs = []
                for d in all_raw_data[name]:
                    iters = d['timestamps'] // AGENTS_PER_STEP
                    val = d['team'] if metric == 'team' else d['length']
                    val_s = smooth_data(val, window=SMOOTH_WINDOW)
                    iter_s = iters[len(iters)-len(val_s):]
                    if len(iter_s) > 1:
                        interp = np.interp(common_x, iter_s, val_s)
                        interp_runs.append(interp)
                        if plot_individual:
                            ax.plot(common_x, interp, color=colors[name], alpha=individual_alpha, linewidth=1)
                if interp_runs:
                    runs_arr = np.array(interp_runs)
                    mean = np.mean(runs_arr, axis=0)
                    p25 = np.percentile(runs_arr, 25, axis=0)
                    p75 = np.percentile(runs_arr, 75, axis=0)
                    ax.plot(common_x, mean, color=colors[name], label=name, linewidth=2)
                    ax.fill_between(common_x, p25, p75, color=colors[name], alpha=0.2)
                    mean_ys[metric].extend(mean)
                    percentile_ys[metric]['p25'].extend(p25)
                    percentile_ys[metric]['p75'].extend(p75)
            
            elif metric == 'archers':
                archer_names = [f"archer_{i}" for i in range(NUM_ARCHERS)]
                for i, archer in enumerate(archer_names):
                    base_color = colors[name]
                    hsv = mcolors.rgb_to_hsv(mcolors.to_rgb(base_color))
                    hsv[0] = (hsv[0] + i * 0.03) % 1
                    agent_color = mcolors.hsv_to_rgb(hsv)
                    interp_runs = []
                    for d in all_raw_data[name]:
                        iters = d['timestamps'] // AGENTS_PER_STEP
                        val_s = smooth_data(d[archer], window=SMOOTH_WINDOW)
                        iter_s = iters[len(iters)-len(val_s):]
                        if len(iter_s) > 1:
                            interp = np.interp(common_x, iter_s, val_s)
                            interp_runs.append(interp)
                            if plot_individual:
                                ax.plot(common_x, interp, color=agent_color, alpha=individual_alpha, linewidth=1)
                    if interp_runs:
                        runs_arr = np.array(interp_runs)
                        mean = np.mean(runs_arr, axis=0)
                        p25 = np.percentile(runs_arr, 25, axis=0)
                        p75 = np.percentile(runs_arr, 75, axis=0)
                        ax.plot(common_x, mean, color=agent_color, label=f"{archer} ({name})", linewidth=2)
                        ax.fill_between(common_x, p25, p75, color=agent_color, alpha=0.2)
                        mean_ys[metric].extend(mean)
                        percentile_ys[metric]['p25'].extend(p25)
                        percentile_ys[metric]['p75'].extend(p75)
            
            elif metric == 'knights':
                knight_names = [f"knight_{i}" for i in range(NUM_KNIGHTS)]
                for i, knight in enumerate(knight_names):
                    base_color = colors[name]
                    hsv = mcolors.rgb_to_hsv(mcolors.to_rgb(base_color))
                    hsv[0] = (hsv[0] + i * 0.03) % 1
                    agent_color = mcolors.hsv_to_rgb(hsv)
                    interp_runs = []
                    for d in all_raw_data[name]:
                        iters = d['timestamps'] // AGENTS_PER_STEP
                        val_s = smooth_data(d[knight], window=SMOOTH_WINDOW)
                        iter_s = iters[len(iters)-len(val_s):]
                        if len(iter_s) > 1:
                            interp = np.interp(common_x, iter_s, val_s)
                            interp_runs.append(interp)
                            if plot_individual:
                                ax.plot(common_x, interp, color=agent_color, alpha=individual_alpha, linewidth=1)
                    if interp_runs:
                        runs_arr = np.array(interp_runs)
                        mean = np.mean(runs_arr, axis=0)
                        p25 = np.percentile(runs_arr, 25, axis=0)
                        p75 = np.percentile(runs_arr, 75, axis=0)
                        ax.plot(common_x, mean, color=agent_color, label=f"{knight} ({name})", linewidth=2)
                        ax.fill_between(common_x, p25, p75, color=agent_color, alpha=0.2)
                        mean_ys[metric].extend(mean)
                        percentile_ys[metric]['p25'].extend(p25)
                        percentile_ys[metric]['p75'].extend(p75)

        ax.legend(loc='upper left')
        ax.set_ylabel("Episode Length" if metric == 'length' else "Raw Reward")
        ax.set_xlabel(f"Iterations (Timesteps / {AGENTS_PER_STEP})")

        # Add agent icons
        move_x, move_y = 0.0, -0.2
        if metric == 'archers':
            try:
                img = mpimg.imread('assets/archer.png')
                img = np.rot90(img, k=-1) # 270 rot
                pos = ax.get_position()
                icon_ax = fig.add_axes([pos.x0 - 0.009 + move_x, pos.y1 + 0.025 + move_y, 0.035, 0.035]) # x left, y bottom, width, height
                icon_ax.imshow(img)
                icon_ax.axis('off')
            except Exception:
                pass
        elif metric == 'knights':
            try:
                img = mpimg.imread('assets/knight.png')
                img = np.rot90(img, 2) # 180 rot
                pos = ax.get_position()
                icon_ax = fig.add_axes([pos.x0 - 0.009 + move_x, pos.y1 - 0.04 + move_y, 0.035, 0.035])
                icon_ax.imshow(img)
                icon_ax.axis('off')
            except Exception:
                pass

    # ylim based on percentiles for all metrics
    for metric in metrics:
        ax = axs[metric_positions[metric]]
        if percentile_ys[metric]['p25']:
            ymin = min(percentile_ys[metric]['p25'])
            ymax = max(percentile_ys[metric]['p75'])
            ax.set_ylim(ymin, ymax)
        elif mean_ys[metric]:
            ymin = min(mean_ys[metric])
            ymax = max(mean_ys[metric])
            ax.set_ylim(ymin, ymax)

    plt.savefig(os.path.join(OUTPUT_DIR, "shaping_comparison.png"), dpi=150)
    print("Plot saved.")
    plt.show()




# ==============
#    PLAYING
# ==============

def run_play(args):
    scenarios = [
        {"name": name, "key": name.replace(' ', '_'), "use_shaping": use_shaping, "shapings": shapings}
        for name, use_shaping, shapings in SCENARIOS
    ]
    
    print("Choose a scenario for playback:")
    n_scenarios = len(scenarios)
    for i, s in enumerate(scenarios, 1):
        print(f"{i}. {s['name']}")
    
    while True:
        try:
            choice = int(input(f"Enter choice (1-{n_scenarios}): "))
            if 1 <= choice <= n_scenarios:
                break
            else:
                print(f"Invalid choice. Enter a number between 1 and {n_scenarios}.")
        except ValueError:
            print("Invalid input. Enter a number.")
    
    selected = scenarios[choice - 1]
    use_shaping = selected["use_shaping"]
    shapings = selected["shapings"]
    scenario_key = selected["key"]
    
    model_path = os.path.join(MODELS_DIR, f"ppo_{scenario_key}_run{args.run}.zip")
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found."); return

    print(f"Loading model for playback: {selected['name']} run {args.run}...")
    model = PPO.load(model_path)

    # manual loop to bypass MarkovVectorEnv errors
    env = knights_archers_zombies_v10.parallel_env(
        render_mode="human", spawn_rate=SPAWN_RATE, 
        num_archers=NUM_ARCHERS, num_knights=NUM_KNIGHTS, vector_state=True
    )
    if use_shaping:
        env = RewardShapingWrapper(env, shapings, total_timesteps=REAL_TOTAL_TIMESTEPS)
    env = ss.black_death_v3(env)
    env = ss.flatten_v0(env)

    obs, infos = env.reset()
    try:
        while env.agents:
            actions = {}
            for agent in env.agents:
                action, _ = model.predict(obs[agent], deterministic=True)
                actions[agent] = action
            
            obs, rewards, terminations, truncations, infos = env.step(actions)
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "plot", "play"])
    parser.add_argument("--individual", action="store_true")
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "train": run_training()
    elif args.mode == "plot": run_plotting(args.individual)
    elif args.mode == "play": run_play(args)