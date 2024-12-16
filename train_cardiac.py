# %%

from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

# %%
#
# Add the rest of the imports here
from dqn_naf.naf import NAF
from dqn_naf.normalized_actions import NormalizedActions
from dqn_naf.ounoise import OUNoise
from dqn_naf.replay_memory import ReplayMemory, Transition
from rl_cardiac.cardiac_model import CardiacModel_Env
from rl_cardiac.tcn_model import TCN_config

# %%
print(torch.cuda.is_available())
# %%

gamma = 0.99
tau = 0.001
ou_noise = True
param_noise = False
noise_scale = 1.0
final_noise_scale = 0.5
exploration_end = 50
seed = 0
batch_size = 128
num_steps = 500
num_episodes = 5000
hidden_size = 32
updates_per_step = 10
replay_size = 10000
t = 2

rat_type = "hypertension_exercise"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# %%
tcn_model = TCN_config(rat_type)
env = NormalizedActions(CardiacModel_Env(tcn_model, rat_type))

# %%
writer = SummaryWriter(log_dir="./runs_cardiac")

# %%
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

# %%
rewards = []
updates = 0
best_rewards = -1e6

df_train = pd.DataFrame()
df_loss = pd.DataFrame()

# %%
for i_episode in range(num_episodes):
    total_numsteps = 0
    state = torch.Tensor([env.reset()[0]])

    if ou_noise:
        ounoise.scale = (noise_scale - final_noise_scale) * max(
            0, exploration_end - i_episode
        ) / exploration_end + final_noise_scale
        ounoise.reset()

    episode_reward = 0

    while True:
        action = agent.select_action(state, ounoise, param_noise)

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

            for _ in range(updates_per_step):
                transitions = memory.sample(batch_size)
                batch = Transition(*zip(*transitions))

                value_loss, policy_loss = agent.update_parameters(batch)

                writer.add_scalar("loss/value", value_loss, updates)
                writer.add_scalar("loss/policy", policy_loss, updates)

                l1 = torch.sum(torch.abs(agent.model.state_dict()["linear1.weight"]))
                l2 = torch.sum(torch.abs(agent.model.state_dict()["linear2.weight"]))
                l3 = torch.sum(torch.abs(agent.model.state_dict()["linear3.weight"]))
                l4 = torch.sum(torch.abs(agent.model.state_dict()["linear4.weight"]))
                writer.add_scalar(
                    "weights/linear1",
                    l1,
                    updates,
                ),
                writer.add_scalar(
                    "weights/linear2",
                    l2,
                    updates,
                ),
                writer.add_scalar(
                    "weights/linear3",
                    l3,
                    updates,
                ),
                writer.add_scalar(
                    "weights/linear4",
                    l4,
                    updates,
                )

                updates += 1

                d = pd.DataFrame(
                    {
                        "updates": [updates],
                        "linear1": [l1],
                        "linear2": [l2],
                        "linear3": [l3],
                        "linear4": [l4],
                        "value_loss": [value_loss],
                    }
                )

                df_loss = pd.concat([df_loss, d], ignore_index=True)

        if done:
            break

    writer.add_scalar("total reward per eposide/train", episode_reward, i_episode)
    writer.add_scalar(
        "Normalized rewards",
        episode_reward / 100,
        i_episode,
    )
    dd = pd.DataFrame(
        {"episode": [i_episode], "reward_per_step": [episode_reward / 100]}
    )

    df_train = pd.concat([df_train, dd], ignore_index=True)

    # Save the weights to the rewards of the best performing model
    if episode_reward > best_rewards:
        best_rewards = episode_reward
        agent.save_model(
            env_name=f"best_rl_cardiac_noise_{str(ou_noise)}_{rat_type}_{timestamp}"
        )

    rewards.append(episode_reward)

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

# %%
df_loss["linear1"] = df_loss["linear1"].apply(lambda x: x.item())
df_loss["linear2"] = df_loss["linear2"].apply(lambda x: x.item())
df_loss["linear3"] = df_loss["linear3"].apply(lambda x: x.item())
df_loss["linear4"] = df_loss["linear4"].apply(lambda x: x.item())

# %%
df_loss.to_csv(
    f"rl_cardiac_best_noise_{str(ou_noise)}_{rat_type}_{timestamp}_loss.csv",
    index=False,
)
df_train.to_csv(
    f"rl_cardiac_best_noise_{str(ou_noise)}_{rat_type}_{timestamp}_rewards.csv",
    index=False,
)
# %%
agent.save_model(
    env_name=f"rl_cardiac_best_noise_{str(ou_noise)}_{rat_type}_{timestamp}"
)
# %%
env.close()
