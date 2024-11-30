# %%
# The necessary imports needed to run the rl-dbs environment
# Convert this script to a jupyter notebook so that it can run on colar
# TODO-  For colab you will have to include the imports like in Yusen's notebook
import numpy as np
from rl_cardiac.tcn_model import TCN_config
from rl_cardiac.cardiac_model import CardiacModel_Env

rat_type = "hypertension_exercise"

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
ou_noise = True
param_noise = False
noise_scale = 0.3
final_noise_scale = 0.3
exploration_end = 50
seed = 0
batch_size = 64
num_steps = 500
num_episodes = 1000
hidden_size = 32
updates_per_step = 10
replay_size = 256
t = 2


# %%
tcn_model = TCN_config(rat_type)
env = NormalizedActions(CardiacModel_Env(tcn_model, rat_type))

# %%
writer = SummaryWriter(log_dir="./runs_cardiac")

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
# Memory
memory = ReplayMemory(replay_size)

# %%
ounoise = OUNoise(env.action_space.shape[0]) if ou_noise else None
param_noise = (
    AdaptiveParamNoiseSpec(
        initial_stddev=0.05,
        desired_action_stddev=noise_scale,
        adaptation_coefficient=1.05,
    )
    if param_noise
    else None
)

# %%
rewards = []
updates = 0

# %%
for i_episode in range(num_episodes):
    total_numsteps = 0
    # print(f"Episode: {i_episode}")
    state = torch.Tensor([env.reset()[0]])

    if ou_noise:
        ounoise.scale = (noise_scale - final_noise_scale) * max(
            0, exploration_end - i_episode
        ) / exploration_end + final_noise_scale
        ounoise.reset()

    episode_reward = 0

    while True and total_numsteps < num_steps:
        # print("Collecting transitions to fill  the memory buffer")
        action = agent.select_action(state, ounoise, param_noise)
        # NOTE - In the previous implementation. They had misnamed the action function
        # This is why the code was not working
        next_state, reward, terminated, truncated, _ = env.step(action.numpy()[0])

        done = terminated or truncated

        total_numsteps += 1
        episode_reward += reward

        action = torch.Tensor(action)
        mask = torch.Tensor([not done])
        next_state = torch.Tensor([next_state])
        reward = torch.Tensor([reward])

        memory.push(state, action, mask, next_state, reward)

        state = next_state

        if len(memory) > batch_size:
            # print(
            #     "Now updating the agent using the collected transitions in the memory buffer"
            # )
            for _ in range(updates_per_step):
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = agent.update_parameters(batch)

                writer.add_scalar("loss/value", value_loss, updates)
                writer.add_scalar("loss/policy", policy_loss, updates)

                updates += 1

        if done:
            break

    # print(f"Episode {i_episode}, Reward {episode_reward}")
    writer.add_scalar("reward/train", episode_reward, i_episode)

    # Update param_noise based on distance metric
    if param_noise:
        episode_transitions = memory.memory[memory.position - t : memory.position]
        states = torch.cat([transition[0] for transition in episode_transitions], 0)
        unperturbed_actions = agent.select_action(states, None, None)
        perturbed_actions = torch.cat(
            [transition[1] for transition in episode_transitions], 0
        )

        ddpg_dist = ddpg_distance_metric(
            perturbed_actions.numpy(), unperturbed_actions.numpy()
        )
        param_noise.adapt(ddpg_dist)

    rewards.append(episode_reward)

    # print("Now testing the trained agent")
    if i_episode % 10 == 0:
        print(f"Episode: {i_episode}")
        state = torch.Tensor([env.reset()[0]])
        episode_reward = 0
        total_numsteps = 0
        while True:
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action.numpy()[0])
            episode_reward += reward
            total_numsteps += 1

            next_state = torch.Tensor([next_state])

            state = next_state

            done = terminated or truncated

            if done:
                break

        writer.add_scalar("reward/test", episode_reward, i_episode)

        rewards.append(episode_reward)
        # print(
        #     "Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(
        #         i_episode, total_numsteps, rewards[-1], np.mean(rewards[-10:])
        #     )
        # )

# %%
agent.save_model(env_name="rl-cardiac")
# %%
env.close()
