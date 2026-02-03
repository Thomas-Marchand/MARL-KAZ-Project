import os
import multiprocessing
from reward_shaping import proximity_shaping, bottom_safety_shaping, position_shaping


TOTAL_TIMESTEPS_MILLIONS = 40
N_RUNS_PER_SCENARIO = 2
PARALLEL_ENVS = 4


TOTAL_TIMESTEPS = TOTAL_TIMESTEPS_MILLIONS * 1_000_000
SPAWN_RATE = 5 # every X steps
NUM_ARCHERS, NUM_KNIGHTS = 2, 2
MAX_ZOMBIES = 10
MAX_ARROWS = 10
N_STEPS_PPO = 2048

N_CPUS = multiprocessing.cpu_count()
# N_CPUS = 1
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
    # ("Base Reward", False, []),
    # ("Proximity Shaping", True, [
    #     {"func": proximity_shaping, "kwargs": {"reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 1.0}
    # ]),
    # ("Bottom Safety Shaping", True, [
    #     {"func": bottom_safety_shaping, "kwargs": {"bottom_threshold": 0.8, "reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 1.0}
    # ]),
    # ("Position Shaping", True, [
    #     {"func": position_shaping, "kwargs": {"n_archers": NUM_ARCHERS, "n_knights": NUM_KNIGHTS, "reward_scale": 0.5}, "start_pct": 0.0, "end_pct": 1.0}
    # ]),
    ("Mixed Shaping_(prox+bottom)", True, [
        {"func": proximity_shaping, "kwargs": {"reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 0.5},
        {"func": bottom_safety_shaping, "kwargs": {"bottom_threshold": 0.8, "reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 0.3}
    ]),
    ("Mixed Shaping_(prox+bottom+pos)", True, [
        {"func": proximity_shaping, "kwargs": {"reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 0.5},
        {"func": bottom_safety_shaping, "kwargs": {"bottom_threshold": 0.8, "reward_scale": 0.05}, "start_pct": 0.0, "end_pct": 0.5},
        {"func": position_shaping, "kwargs": {"n_archers": NUM_ARCHERS, "n_knights": NUM_KNIGHTS, "reward_scale": 0.5}, "start_pct": 0.0, "end_pct": 0.3},
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