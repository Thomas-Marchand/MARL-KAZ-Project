import os
import sys
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import supersuit as ss
import multiprocessing
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.utils import set_random_seed
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.wrappers import BaseParallelWrapper

# CONFIG
TOTAL_TIMESTEPS = 5000000
NUM_RUNS = 8

SPAWN_RATE = 5 # every X steps
NUM_ARCHERS = 2
NUM_KNIGHTS = 2
MAX_ZOMBIES = 10
MAX_ARROWS = 10
PARALLEL_ENVS = 8
TOTAL_AGENTS_PER_STEP = (NUM_ARCHERS + NUM_KNIGHTS) * PARALLEL_ENVS
N_STEPS_PPO = 2048

STEP_SIZE = N_STEPS_PPO * TOTAL_AGENTS_PER_STEP
REAL_TOTAL_TIMESTEPS = ((TOTAL_TIMESTEPS + STEP_SIZE - 1) // STEP_SIZE) * STEP_SIZE

OUTPUT_DIR = f"./reward_shaping_experiment_{PARALLEL_ENVS}x{NUM_RUNS}x{REAL_TOTAL_TIMESTEPS//1000000}M/"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

AGENT_NAMES = [f"archer_{i}" for i in range(NUM_ARCHERS)] + [f"knight_{i}" for i in range(NUM_KNIGHTS)]

print(f"Output Directory: {OUTPUT_DIR}")
print(f"Total Timesteps per run: {REAL_TOTAL_TIMESTEPS} ~ {REAL_TOTAL_TIMESTEPS//1000000}M")

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

WINDOW = 100 # Don't choose a multiple of 8 to avoid artifacts !

# REWARD SHAPING WRAPPER
class KnightShapingWrapper(BaseParallelWrapper):
    """
    Adds a reward to knights based on proximity to the nearest zombie.
    Observation structure (vector_state=True):
    Zombie rows start at index 18. Column 0 is the distance.
    """
    def step(self, actions):
        obs, rewards, terminations, truncations, infos = super().step(actions)
        
        z_start, z_end = 18, 28 

        for agent in self.agents:
            shaped_val = 0.0
            if "knight" in agent and agent in obs:
                zombie_dists = obs[agent][z_start:z_end, 0]
                valid_dists = zombie_dists[zombie_dists > 0]
                
                if len(valid_dists) > 0:
                    closest_dist = np.min(valid_dists)
                    # Reward: up to 0.05 per step for being near a zombie
                    shaped_val = 0.05 * (1.0 - closest_dist)
                    original_reward = float(rewards[agent])
                    rewards[agent] = original_reward + shaped_val
            
            # Store shaping info in the info dict for the callback to see
            if agent in infos:
                infos[agent]["shaping_reward"] = shaped_val
                infos[agent]["original_reward"] = original_reward if "knight" in agent and len(valid_dists) > 0 else float(rewards[agent])
        
        # Set for dead agents
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


def make_env(use_shaping, render_mode=None, num_envs=8):
    env = knights_archers_zombies_v10.parallel_env(
        spawn_rate=SPAWN_RATE, num_archers=NUM_ARCHERS, num_knights=NUM_KNIGHTS,
        max_zombies=MAX_ZOMBIES, max_arrows=MAX_ARROWS,
        vector_state=True, use_typemasks=False, render_mode=render_mode
    )
    if use_shaping:
        env = KnightShapingWrapper(env)
    env = ss.black_death_v3(env)
    env = ss.flatten_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    if num_envs > 1:
        env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=0, base_class='stable_baselines3')
    env = VecMonitor(env)
    return env


# MODES: TRAIN / PLOT / PLAY

def train_single_run(args):
    name, use_shaping, run_idx = args
    run_name = f"{name.replace(' ', '_')}_run{run_idx}"
    set_random_seed(run_idx * 42)
    
    print(f"STARTING: {name} | RUN: {run_idx}")
    start_time = time.time()
    env = make_env(use_shaping=use_shaping, num_envs=PARALLEL_ENVS)
    model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, batch_size=2048)
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
    scenarios = [("Shaped Reward", True), ("Base Reward", False)]
    tasks = [(name, use_shaping, i) for name, use_shaping in scenarios for i in range(1, NUM_RUNS + 1)]

    print(f"Starting Multi-processing training for {len(tasks)} runs, {REAL_TOTAL_TIMESTEPS} timesteps each...")
    with multiprocessing.Pool(processes=8) as pool:
        pool.map(train_single_run, tasks)



# MODE: PLOT
def smooth_data(values, window=20):
    if len(values) < window: return values
    return np.convolve(values, np.ones(window)/window, mode='valid')

def run_plotting(plot_individual=False):
    print(f"\nGenerating plots from logs, plot_individual={plot_individual}...")
    scenarios = ["Shaped_Reward", "Base_Reward"]
    colors = {"Shaped_Reward": "#2a9dff", "Base_Reward": "#ff5353"}

    all_raw_data = {s: [] for s in scenarios}
    runtimes = {s: [] for s in scenarios}
    max_iter = 0
    min_iter = float('inf')

    for name in scenarios:
        files = glob.glob(os.path.join(LOG_DIR, f"stats_{name}_run*.npz"))
        for f in files:
            d = np.load(f)
            iters = d['timestamps'] // TOTAL_AGENTS_PER_STEP
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
    suptitle_parts = [f"{NUM_RUNS} runs each"]
    for name in scenarios:
        suptitle_parts.append(f"{name}: {scenario_stats[name]['episodes']} episodes, avg {scenario_stats[name]['avg_runtime']:.1f}s")
    suptitle_parts.append(f"Smoothing window={WINDOW}")
    suptitle = "Training Analysis: " + ", ".join(suptitle_parts)
    fig.suptitle(suptitle, fontsize=12, fontweight='bold')

    mean_ys = {metric: [] for metric in metrics}
    percentile_ys = {metric: {'p25': [], 'p75': []} for metric in metrics}

    for metric in metrics:
        ax = axs[metric_positions[metric]]
        
        # Title
        base_title = metric.capitalize()
        title_info = f"{base_title} (25-75% percentiles)"
        ax.set_title(title_info, fontsize=14, fontweight='bold')
        
        for name in scenarios:
            if metric in ['team', 'length']:
                interp_runs = []
                for d in all_raw_data[name]:
                    iters = d['timestamps'] // TOTAL_AGENTS_PER_STEP
                    val = d['team'] if metric == 'team' else d['length']
                    val_s = smooth_data(val, window=WINDOW)
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
                    hsv[0] = (hsv[0] + i * 0.05) % 1
                    agent_color = mcolors.hsv_to_rgb(hsv)
                    interp_runs = []
                    for d in all_raw_data[name]:
                        iters = d['timestamps'] // TOTAL_AGENTS_PER_STEP
                        val_s = smooth_data(d[archer], window=WINDOW)
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
                    hsv[0] = (hsv[0] + i * 0.1) % 1
                    agent_color = mcolors.hsv_to_rgb(hsv)
                    interp_runs = []
                    for d in all_raw_data[name]:
                        iters = d['timestamps'] // TOTAL_AGENTS_PER_STEP
                        val_s = smooth_data(d[knight], window=WINDOW)
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

        # ax.legend(loc='upper right')
        ax.legend(loc='upper left')
        ax.set_ylabel("Episode Length" if metric == 'length' else "Raw Reward")
        ax.set_xlabel(f"Iterations (Timesteps / {TOTAL_AGENTS_PER_STEP})")

        # Add agent icons
        move_x, move_y = 0.0, -0.1
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

def run_play(args):
    use_shaping = (args.scenario == "Shaped_Reward")
    model_path = os.path.join(MODELS_DIR, f"ppo_{args.scenario}_run{args.run}.zip")
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found."); return

    print(f"Loading model for playback: {args.scenario} run {args.run}...")
    model = PPO.load(model_path)

    # manual loop to bypass MarkovVectorEnv errors
    env = knights_archers_zombies_v10.parallel_env(
        render_mode="human", spawn_rate=SPAWN_RATE, 
        num_archers=NUM_ARCHERS, num_knights=NUM_KNIGHTS, vector_state=True
    )
    if use_shaping:
        env = KnightShapingWrapper(env)
    env = ss.black_death_v3(env)
    env = ss.flatten_v0(env)

    obs, infos = env.reset()
    try:
        while env.agents:
            actions = {}
            for agent in env.agents:
                # model.predict handles raw flattened vector from PettingZoo ?
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
    parser.add_argument("--scenario", choices=["Shaped_Reward", "Base_Reward"], default="Shaped_Reward")
    parser.add_argument("--run", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "train": run_training()
    elif args.mode == "plot": run_plotting(args.individual)
    elif args.mode == "play": run_play(args)