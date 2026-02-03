import os
import multiprocessing

TOTAL_TIMESTEPS_MILLIONS = 10
N_RUNS_PER_SCENARIO = 2
PARALLEL_ENVS = 2

TOTAL_TIMESTEPS = TOTAL_TIMESTEPS_MILLIONS * 1_000_000
SPAWN_RATE = 5
NUM_ARCHERS, NUM_KNIGHTS = 2, 2
MAX_ZOMBIES = 10
MAX_ARROWS = 10
N_STEPS_PPO = 2048

N_CPUS = multiprocessing.cpu_count()
# N_CPUS = 2 
AGENTS_PER_STEP = (NUM_ARCHERS + NUM_KNIGHTS) * PARALLEL_ENVS
STEP_SIZE = N_STEPS_PPO * AGENTS_PER_STEP
REAL_TOTAL_TIMESTEPS = ((TOTAL_TIMESTEPS + STEP_SIZE - 1) // STEP_SIZE) * STEP_SIZE

# COLORS for scenarios
COLORS = {
    "Shared_Vision_0.2": "#2a9dff",       # Blue (Centralized/Team)
    "Decentralized_0.2": "#ff5353",       # Red (Individual)
    "Full_Vision": "#808080"        # Gray (Control)
}

# SCENARIOS: (name, observation_radius, use_shaping, shapings_list)
SCENARIOS = [
    ("Decentralized_0.2", "decentralized", 0.2, False, []),
    ("Shared_Vision_0.2", "shared", 0.2, False, []),
    ("Full_Vision", "full", 100.0, False, []) 
]

OUTPUT_DIR = f"./VisionCompare_{PARALLEL_ENVS}x{N_RUNS_PER_SCENARIO}x{REAL_TOTAL_TIMESTEPS//1000}K/"
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

AGENT_NAMES = [f"archer_{i}" for i in range(NUM_ARCHERS)] + [f"knight_{i}" for i in range(NUM_KNIGHTS)]

print(f"Output Directory: {OUTPUT_DIR}")
print(f"Total Timesteps per run: {REAL_TOTAL_TIMESTEPS} ~ {REAL_TOTAL_TIMESTEPS//1000}K")

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)

SMOOTH_WINDOW = 2000 if REAL_TOTAL_TIMESTEPS >= 20_000_000 else 500 if REAL_TOTAL_TIMESTEPS >= 1_000_000 else 100