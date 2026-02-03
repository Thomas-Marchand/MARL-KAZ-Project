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
from config import (TOTAL_TIMESTEPS, N_RUNS_PER_SCENARIO, SPAWN_RATE, NUM_ARCHERS, NUM_KNIGHTS, MAX_ZOMBIES, MAX_ARROWS,
                    PARALLEL_ENVS, N_STEPS_PPO, REAL_TOTAL_TIMESTEPS, AGENT_NAMES, LOG_DIR, MODELS_DIR, SCENARIOS, N_CPUS)


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
        NUM_AGENTS = NUM_ARCHERS + NUM_KNIGHTS
        if not infos: return True

        # Accumulate shaping rewards per agent
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
                    print(f"Run {self.run_idx:<2}: {pct:5.1f}% | True Rew: [{true_str}] | Tot: {true_team:4.0f} | Len: {ep_stat['l']:<4} | Shaping: [{shaping_str}]")
                
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
    callback = AgentBreakdownCallback(run_idx, use_shaping=use_shaping)
    if name == "Random":
        obs = env.reset() # random logic
        callback.init_callback(model=None)
        current_steps = 0
        while current_steps < REAL_TOTAL_TIMESTEPS:
            # random actions for all parallel envs and all agents
            actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
            obs, rewards, dones, infos = env.step(actions)
            
            callback.locals = {"infos": infos}
            callback.num_timesteps = current_steps
            callback._on_step()
            
            current_steps += PARALLEL_ENVS * (NUM_ARCHERS + NUM_KNIGHTS)
    else:
        model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, batch_size=2048, n_steps=N_STEPS_PPO)
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
        model.save(os.path.join(MODELS_DIR, f"ppo_{run_name}"))
    
    # model = PPO("MlpPolicy", env, verbose=0, learning_rate=3e-4, batch_size=2048, n_steps=N_STEPS_PPO)
    # model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    runtime = time.time() - start_time
    data = callback.get_data()
    data['runtime'] = runtime

    np.savez(os.path.join(LOG_DIR, f"stats_{run_name}.npz"), **data)
    # model.save(os.path.join(MODELS_DIR, f"ppo_{run_name}"))
    env.close()
    
    print(f"FINISHED: {name} | RUN: {run_idx} | Runtime: {runtime:.2f}s")
    return True

def run_training():
    scenarios = SCENARIOS
    tasks = [(name, use_shaping, shapings, i) for name, use_shaping, shapings in scenarios for i in range(1, N_RUNS_PER_SCENARIO + 1)]

    print(f"Starting Multi-processing training for {len(tasks)} runs, {REAL_TOTAL_TIMESTEPS} timesteps each...")
    with multiprocessing.Pool(processes=N_CPUS) as pool:
        pool.map(train_single_run, tasks)


if __name__ == "__main__":
    run_training()