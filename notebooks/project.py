# %%
# Matplotlib Inline
import matplotlib.pyplot as plt

# %matplotlib inline

# %%
# Agent Framework
import numpy as np
import gym


class BaseAgent:
    def __init__(self, env, verbose=1, ran_seed=42):
        self.env = env
        # random seed is only set once when the agent is initialized
        self.env.seed(ran_seed)
        self.env.action_space.seed(ran_seed + 1)  # why isnt this set at env.seed?
        self.env.observation_space.seed(ran_seed + 2)
        self.random_state = np.random.RandomState(ran_seed + 3)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.verbose = verbose
        self.cumulative_reward = 0
        self.num_steps = 0

    def select_action(self, state):
        raise NotImplementedError

    def update_step(self, reward: float):
        self.cumulative_reward += reward
        self.num_steps += 1

    def update_episode(self):
        self.reset_episode()

    def update_rollout(self):
        if self.verbose > 0:
            print("update_rollout in base class is called, nothing is changed")

    def update_replay(self):
        if self.verbose > 0:
            print("update_replay in base class is called, nothing is changed")

    def reset_episode(self):
        self.cumulative_reward = 0
        self.num_steps = 0


class RandomAgent(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.cumulative_reward = 0
        super().__init__(*args, **kwargs)

    def select_action(self, state):
        action = self.action_space.sample()
        if self.verbose > 1:
            print("Random agent selected action: ", action)
        return action

    def update_step(self, old_state, action, reward, new_state):
        super().update_step(reward)

    def update_episode(self, terminated, truncated):
        if self.verbose > 0:
            if terminated:
                print("Episode terminated")
            if truncated:
                print("Episode truncated")
        super().update_episode()

    def update_rollout(self):
        pass

    def update_replay(self):
        pass


# %%
# Example usage of rl cardiac
from rl_cardiac.tcn_model import TCN_config
from rl_cardiac.cardiac_model import CardiacModel_Env
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.logger import configure


# log_dir = r"C:/Users/timot/OneDrive/Desktop/EMORY/Fall2024/BMI539_RL/Course_Project/github/bmi539-rl/results"
# os.makedirs(log_dir, exist_ok=True)

rat_type = "healthy_stable"

tcn_model = TCN_config(rat_type=rat_type)
env = CardiacModel_Env(tcn_model=tcn_model, rat_type=rat_type)
# noise level is set to 0 by default, should be changed to see if your agent can handle noise once it works well without noise
# env = CardiacModel_Env(tcn_model, rat_type, noise_level)

# %%

# env = Monitor(env, log_dir)
env = Monitor(env)


log_dir = "./logs/"
new_logger = configure(log_dir, ["stdout", "tensorboard"])


# %%
from stable_baselines3 import PPO

policy_kwargs = dict(net_arch=[64])
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.002,
    n_steps=128,
    batch_size=4,
    n_epochs=4,
    clip_range=0.2,
    gamma=0.95,
    vf_coef=1,
    ent_coef=0.005,
    policy_kwargs=policy_kwargs,
)
env.seed = 42
env.reset()
model.learn(total_timesteps=5000)
model.set_logger(new_logger)
# %%

import pandas as pd
import matplotlib.pyplot as plt

# Load monitor.csv file (replace with the path to your actual file)
data = pd.read_csv(
    r"/mnt/c/Users/timot/OneDrive/Desktop/EMORY/Fall2024/BMI539_RL/Course_Project/github/bmi539-rl/c:/Users/timot/OneDrive/Desktop/EMORY/Fall2024/BMI539_RL/Course_Project/github/bmi539-rl/results/monitor.csv",
    skiprows=1,
)
timesteps = data["t"]  # Cumulative timesteps
rewards = data["r"]  # Episode rewards

# Plotting the learning curve
plt.plot(timesteps, rewards)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Learning Curve")
plt.show()
# %%
# Performing Predictions on the trained model
obs = env.reset()[0]
r = []
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    r.append(rewards)
# %%
plt.plot(r)

# %%
# Example usage of the rl dbs
import gymnasium as gym
import rl_dbs.gym_oscillator
import rl_dbs.gym_oscillator.envs
import rl_dbs.oscillator_cpp

# from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# import gym

# from stable_baselines3.common.env_util import make_vec_env

env = rl_dbs.gym_oscillator.envs.oscillatorEnv()

# env = gym.make("oscillator-v0")

# %%
# log_dir = r"C:/Users/timot/OneDrive/Desktop/EMORY/Fall2024/BMI539_RL/Course_Project/github/bmi539-rl/results_two"
# os.makedirs(log_dir, exist_ok=True)

# Configure TensorBoard logging
log_dir = "./logs/"
new_logger = configure(log_dir, ["stdout", "tensorboard"])

# %%
# env = Monitor(env, log_dir)
# env = Monitor(env, filename="results.csv")
env = Monitor(env)
# %%
from stable_baselines3 import PPO

policy_kwargs = dict(net_arch=[64])
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.002,
    n_steps=128,
    batch_size=4,
    n_epochs=4,
    clip_range=0.2,
    gamma=0.95,
    vf_coef=1,
    ent_coef=0.005,
    policy_kwargs=policy_kwargs,
)
model.set_logger(new_logger)
env.seed = 42
env.reset()
model.learn(total_timesteps=50000)

# %%
env.close()

# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load monitor.csv file (replace with the path to your actual file)
data = pd.read_csv(
    r"results.csv",
    skiprows=1,
)
timesteps = data["t"]  # Cumulative timesteps
rewards = data["r"]  # Episode rewards

# Plotting the learning curve
plt.plot(timesteps, rewards)
plt.xlabel("Timesteps")
plt.ylabel("Episode Reward")
plt.title("Learning Curve")
plt.show()

# %%
# Performing Predictions on the trained model
obs = env.reset()[0]
r = []
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    r.append(rewards)

    if truncated:
        break


# %%
plt.plot(r)

# %%
