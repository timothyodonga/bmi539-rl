# %%
# The necessary imports needed to run the rl-dbs environment
# Convert this script to a jupyter notebook so that it can run on colar
# TODO-  Add patience to the code so that you stop one the best rewards do not increase past a certain points in the training

import gymnasium as gym
import rl_dbs.gym_oscillator
import rl_dbs.gym_oscillator.envs
import rl_dbs.oscillator_cpp
import numpy as np

# env = gym.make('oscillator-v0')

# env = rl_dbs.gym_oscillator.envs.oscillatorEnv()

# %%
# Imports from the NAF implementation
import argparse
import math
from collections import namedtuple
from itertools import count

import gym
import numpy as np
import torch
from gym import wrappers

# %%
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# %%
#
# Add the rest of the imports here
from dqn_naf.naf import NAF
from dqn_naf.normalized_actions import NormalizedActions
from dqn_naf.ounoise import OUNoise
from dqn_naf.param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
from dqn_naf.replay_memory import ReplayMemory, Transition

print(torch.cuda.is_available())
# %%
# TODO - Update these values to suit the rlb-dbs problem
gamma = 0.99
tau = 0.001
ou_noise = False
param_noise = False
noise_scale = 0.3
final_noise_scale = 0.3
exploration_end = 50
seed = 0
batch_size = 64
num_steps = 1000
num_episodes = 1000
hidden_size = 32
updates_per_step = 50
replay_size = 256
t = 2


# %%
# env = NormalizedActions(rl_dbs.gym_oscillator.envs.oscillatorEnv())
env = rl_dbs.gym_oscillator.envs.oscillatorEnv()

# %%
writer = SummaryWriter()

# %%
# env.seed(seed) #TODO - Double check this
torch.manual_seed(seed)
np.random.seed(seed)

# %%
# Create the agent
agent = NAF(
    gamma,
    tau,
    hidden_size,
    env.observation_space.shape[0],
    env.action_space,
)
# %%
# %%
# Load the best agent saved
agent.load_model(model_path=r"models/naf_best_rl_dbs_.pth")
# %%
env.reset()
state_list = []
actions_list = []
state = torch.Tensor([env.reset()[0]])
rewards_list = []

for i in range(10000):
    # action = agent.select_action(state)
    next_state, reward, terminated, truncated, _ = env.step(torch.Tensor([0]))
    state = torch.Tensor([next_state])
    state_list.append(state)
    actions_list.append(np.array([0]))
# %%
state_list_agent = []
actions_list_agent = []

for i in range(10000):
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action.numpy()[0])

    state = torch.Tensor([next_state])

    state_list_agent.append(state)
    actions_list_agent.append(action.numpy()[0])
    rewards_list.append(reward)

    # done = terminated or truncated

    # if done:
    #     break

# %%
env.close()

# %%
full_state_list = state_list + state_list_agent
full_action_list = actions_list + actions_list_agent
# %%
s = np.vstack(full_state_list)
s_ = s.mean(axis=1)
a = np.vstack(full_action_list)
a_ = a.squeeze()
# %%
# Plotting the values state and actions of the agent
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rcParams["figure.figsize"] = [16, 10]
plt.rcParams.update({"font.size": 16})

# %%
plt.figure()
plt.plot(
    s[:, -1],  # Plot the value of the last neuron
    linestyle="-",
    label=f"State",
)
plt.plot(
    a_,
    linestyle="-",
    label=f"Action",
)
plt.xlabel("t")
plt.ylabel("X(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"state_action_steps_one_episode_.png")
plt.show()

# %%
