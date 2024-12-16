# %%
import pandas as pd

# %%
df_healthy_stable = pd.read_csv(
    "rl_cardiac_best_noise_True_healthy_stable_20241215_223423_rewards.csv"
)

# %%
df_healthy_stable.max()
# %%
df_hypertension_stable = pd.read_csv(
    "rl_cardiac_best_noise_True_hypertension_stable_20241215_212543_rewards.csv"
)

# %%
df_hypertension_stable.max()
# %%
