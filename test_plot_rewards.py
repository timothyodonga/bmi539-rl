# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

path = "Dec14_11-15-30_todonga.csv"


# %%
def ewma_smoothing(df, column, alpha=0.9):
    """Applies EWMA smoothing to a given column in a DataFrame."""
    return df[column].ewm(alpha=1 - alpha).mean()


# %%
# Get the downloaded file of the rewards
df = pd.read_csv(path)
df["smoothened_value"] = ewma_smoothing(df, "Value", alpha=0.6)
# %%
plt.rcParams["figure.figsize"] = [16, 10]
plt.rcParams.update({"font.size": 16})

# %%
plt.figure()
plt.plot(
    df["Step"],
    df["Value"],
    color="k",
    linestyle="-",
    label=f"Raw rewards",
)
plt.plot(
    df["Step"],
    df["smoothened_value"],
    color="royalblue",
    linewidth=4,
    linestyle="-",
    label=f"Rewards, smoothing factor = 0.6",
)
plt.xticks(range(0, 91, 10))  # Tick marks from 0 to 100 in increments of 10
plt.xlabel("Episodes")
plt.ylabel("Reward per step")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"rl-dbs-avg-rewards.png")
plt.show()

# %%
