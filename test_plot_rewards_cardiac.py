# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# path = "rl_cardiac_best_noise_True_hypertension_stable_20241215_212543_rewards.csv"
# path_loss = "rl_cardiac_best_noise_True_hypertension_stable_20241215_212543_loss.csv"

path = "rl_cardiac_best_noise_True_healthy_stable_20241215_223423_rewards.csv"
path_loss = "rl_cardiac_best_noise_True_healthy_stable_20241215_223423_loss.csv"


# %%
def ewma_smoothing(df, column, alpha=0.9):
    """Applies EWMA smoothing to a given column in a DataFrame."""
    return df[column].ewm(alpha=1 - alpha).mean()


# %%
# Get the downloaded file of the rewards
df = pd.read_csv(path)
df["smoothened_value"] = ewma_smoothing(df, "reward_per_step", alpha=0.6)
# %%
plt.rcParams["figure.figsize"] = [16, 10]
plt.rcParams.update({"font.size": 16})

# %%
plt.figure()
plt.plot(
    df["episode"],
    df["reward_per_step"],
    color="k",
    linestyle="-",
    label=f"Raw rewards",
)
plt.plot(
    df["episode"],
    df["smoothened_value"],
    color="royalblue",
    linewidth=4,
    linestyle="-",
    label=f"Rewards, smoothing factor = 0.6",
)
plt.ylim(0, 1)  # Min value of -5, Max value of 110
plt.xticks(range(0, 91, 10))  # Tick marks from 0 to 100 in increments of 10
plt.xlabel("Episodes")
plt.ylabel("Reward per step")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig(
#     f"C:/Users/timot/OneDrive/Desktop/EMORY/Fall2024/BMI539_RL/Project_Report/report/figures/rewards_healthy_stable.png"
# )
plt.savefig(f"rewards_healthy_stable.png")
plt.show()

# %%
df_loss = pd.read_csv(path_loss)

# %%
plt.figure()
plt.plot(
    df_loss["updates"],
    df_loss["linear1"],
    linestyle="-",
    label=f"linear1",
)
plt.plot(
    df_loss["updates"],
    df_loss["linear2"],
    linestyle="-",
    label=f"linear2",
)
plt.plot(
    df_loss["updates"],
    df_loss["linear3"],
    linestyle="-",
    label=f"linear3",
)
plt.plot(
    df_loss["updates"],
    df_loss["linear4"],
    linestyle="-",
    label=f"linear4",
)
plt.xlabel("Iterations")
# plt.ylabel("")
plt.legend()
plt.grid(True)
plt.tight_layout()
# plt.savefig(
#     f"C:/Users/timot/OneDrive/Desktop/EMORY/Fall2024/BMI539_RL/Project_Report/report/figures/weights_healthy_stable.png"
# )
plt.savefig(f"weights_healthy_stable.png")
plt.show()

# %%
