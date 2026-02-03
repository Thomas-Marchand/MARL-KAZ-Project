import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mcolors
import os
import glob
from config_vectors import AGENTS_PER_STEP, LOG_DIR, OUTPUT_DIR, COLORS, N_RUNS_PER_SCENARIO, NUM_ARCHERS, NUM_KNIGHTS, SCENARIOS, SMOOTH_WINDOW

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
    scenario_names = [name.replace(' ', '_').replace('(', '').replace(')', '') for name, _, _, _, _ in SCENARIOS]
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

if __name__ == "__main__":
    run_plotting(plot_individual=True)