import os
from pettingzoo.butterfly import knights_archers_zombies_v10
from config_vectors import NUM_ARCHERS, NUM_KNIGHTS, SPAWN_RATE, SCENARIOS, MODELS_DIR
from stable_baselines3 import PPO
import supersuit as ss

def run_play(num):
    scenarios = [
        {"name": name, "key": name.replace(' ', '_').replace('(', '').replace(')', ''), "vision_mode": vision_mode, "obs_radius": obs_radius, "use_shaping": use_shaping, "shapings": shapings}
        for name, vision_mode, obs_radius, use_shaping, shapings in SCENARIOS
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
    scenario_key = selected["key"]
    
    model_path = os.path.join(MODELS_DIR, f"ppo_{scenario_key}_run{num}.zip")
    if not os.path.exists(model_path):
        print(f"Model {model_path} not found."); return

    print(f"Loading model for playback: {selected['name']} run {num}...")
    model = PPO.load(model_path)

    # manual loop to bypass MarkovVectorEnv errors
    env = knights_archers_zombies_v10.parallel_env(
        render_mode="human", spawn_rate=SPAWN_RATE, 
        num_archers=NUM_ARCHERS, num_knights=NUM_KNIGHTS, vector_state=True
    )
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
    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    run_play(1)