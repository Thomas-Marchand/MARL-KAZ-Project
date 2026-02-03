import os
import numpy as np
import supersuit as ss
import multiprocessing
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.utils import set_random_seed
from pettingzoo.butterfly import knights_archers_zombies_v10
from pettingzoo.utils.wrappers import BaseParallelWrapper
from config_vectors import (TOTAL_TIMESTEPS, N_RUNS_PER_SCENARIO, SPAWN_RATE, NUM_ARCHERS, NUM_KNIGHTS, MAX_ZOMBIES, MAX_ARROWS,
                    PARALLEL_ENVS, N_STEPS_PPO, REAL_TOTAL_TIMESTEPS, AGENT_NAMES, LOG_DIR, MODELS_DIR, SCENARIOS, N_CPUS)


class DecentralizedVisionWrapper(BaseParallelWrapper):
    """ Agent X only sees entities that are within 'radius' of Agent X """
    def __init__(self, env, radius=0.5):
        super().__init__(env)
        self.radius = radius
        self.step_count = 0

    def mask_observation(self, obs):
        # Column 0 is distance to self.
        # Mask rows where distance > radius
        mask = obs[:, 0] > self.radius
        obs[mask] = 0.0
        return obs

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = super().step(actions)
        for agent in obs:
            obs[agent] = self.mask_observation(obs[agent])
        self.step_count += 1
        if self.step_count % 1000 == 0:
            for agent in obs:
                num_entities = np.sum(np.any(obs[agent] != 0, axis=1))
                # print(f"Vision: decentralized, Range: {self.radius}, Agent: {agent}, Entities seen: {num_entities}")
        return obs, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        for agent in obs:
            obs[agent] = self.mask_observation(obs[agent])
        return obs, infos


class SharedVisionWrapper(BaseParallelWrapper):
    """  Agent X sees an entity if ANY agent is within 'radius' of that entity """
    def __init__(self, env, radius=0.5):
        super().__init__(env)
        self.radius = radius
        self.step_count = 0

    def step(self, actions):
        obs, rewards, terminations, truncations, infos = super().step(actions)
        obs = self.apply_shared_mask(obs)
        self.step_count += 1
        if self.step_count % 1000 == 0:
            agent = list(obs.keys())[0]
            num_entities = np.sum(np.any(obs[agent] != 0, axis=1))
            # print(f"Vision: shared, Range: {self.radius}, Entities seen: {num_entities}")
        return obs, rewards, terminations, truncations, infos

    def reset(self, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)
        obs = self.apply_shared_mask(obs)
        return obs, infos

    def apply_shared_mask(self, obs_dict):
        if not obs_dict:
            return obs_dict
            
        first_agent_obs = next(iter(obs_dict.values()))
        num_entities = first_agent_obs.shape[0] # supposition: all agents have same obs shape
        
        global_hidden_mask = np.ones(num_entities, dtype=bool) # start with all True (hidden)

        for agent_obs in obs_dict.values():
            # dist is col 0, we check if > radius
            agent_specific_hidden = agent_obs[:, 0] > self.radius
            # if global AND agent is hidden, remains hidden, else it's visible for at least 1 agent
            global_hidden_mask = np.logical_and(global_hidden_mask, agent_specific_hidden)

        for agent in obs_dict:
            # hiding the entities no one sees
            obs_dict[agent][global_hidden_mask] = 0.0
            
        return obs_dict


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
        NUM_AGENTS = NUM_ARCHERS + NUM_KNIGHTS
        if not infos: return True

        for i, info in enumerate(infos):
            env_idx = i // NUM_AGENTS
            agent_idx = i % NUM_AGENTS
            agent_name = AGENT_NAMES[agent_idx]
            
            shaped = info.get("shaping_reward", 0)
            self.current_shaping[agent_name] += shaped
            original = info.get("original_reward", 0)
            self.current_original[env_idx][agent_name] += original
        
        for env_idx in range(0, len(infos), NUM_AGENTS):
            env_num = env_idx // NUM_AGENTS
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
                    r = infos[actual_idx]["episode"]["r"] if "episode" in infos[actual_idx] else 0
                    if self.use_shaping:
                        true_r = self.current_original[env_num][agent_name]
                    else:
                        true_r = r 
                    self.history[agent_name].append(true_r)
                    team_reward += r
                    shaping_total += self.current_shaping[agent_name]
                    true_team += true_r
                    true_rewards.append(true_r)
                
                self.history['team'].append(true_team)
                self.history['shaping_sum'].append(shaping_total)
                self.history['true_team'].append(true_team)

                if env_idx == 0:
                    true_str = ", ".join([f"{r:>4.0f}" for r in true_rewards])
                    pct = (current_ts/REAL_TOTAL_TIMESTEPS)*100
                    print(f"Run {self.run_idx:<2}: {pct:5.1f}% | True Rew: [{true_str}] | Tot: {true_team:4.0f} | Len: {ep_stat['l']:<4}")
                
                self.current_shaping = {name: 0.0 for name in AGENT_NAMES}
                self.current_original[env_num] = {name: 0.0 for name in AGENT_NAMES}
        return True

    def get_data(self):
        return {k: np.array(v) for k, v in self.history.items()}


def make_env(vision_mode, obs_radius, use_shaping, render_mode=None, num_envs=8, shapings=None, total_timesteps=None):
    env = knights_archers_zombies_v10.parallel_env(
        spawn_rate=SPAWN_RATE, num_archers=NUM_ARCHERS, num_knights=NUM_KNIGHTS,
        max_zombies=MAX_ZOMBIES, max_arrows=MAX_ARROWS,
        vector_state=True, use_typemasks=False, render_mode=render_mode
    )
    
    if vision_mode == "shared":
        env = SharedVisionWrapper(env, radius=obs_radius)
    elif vision_mode == "decentralized":
        env = DecentralizedVisionWrapper(env, radius=obs_radius)
    else:
        pass # no wrapper

    # 3. Standard PettingZoo Wrappers
    env = ss.black_death_v3(env)
    env = ss.flatten_v0(env)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    if num_envs > 1:
        env = ss.concat_vec_envs_v1(env, num_envs, num_cpus=0, base_class='stable_baselines3')
    env = VecMonitor(env)
    return env


def train_single_run(args):
    name, vision_mode, obs_radius, use_shaping, shapings, run_idx = args
    
    run_name = f"{name.replace(' ', '_')}_run{run_idx}"
    set_random_seed(run_idx * 42)
    
    print(f"STARTING: {name} (Mode: {vision_mode}, R={obs_radius}) | RUN: {run_idx}")
    start_time = time.time()
    
    env = make_env(
        vision_mode=vision_mode,
        obs_radius=obs_radius,
        use_shaping=use_shaping, 
        num_envs=PARALLEL_ENVS, 
        shapings=shapings, 
        total_timesteps=REAL_TOTAL_TIMESTEPS
    )
    
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
    tasks = []
    # config needs to match the unpack in train_single_run
    for (name, vision_mode, obs_radius, use_shaping, shapings) in scenarios:
        for i in range(1, N_RUNS_PER_SCENARIO + 1):
            tasks.append((name, vision_mode, obs_radius, use_shaping, shapings, i))

    print(f"Starting Multi-processing training for {len(tasks)} runs...")
    with multiprocessing.Pool(processes=N_CPUS) as pool:
        pool.map(train_single_run, tasks)

if __name__ == "__main__":
    run_training()