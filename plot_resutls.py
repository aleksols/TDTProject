import matplotlib.pyplot as plt
import pandas as pd
import os

# experiment = "experiments/1639296268.9138777"
experiment = "experiments/100_epoch_ppo1639398377.608867"
experiment = "experiments/100_epoch_ppo1639491161.8668485"
experiment = "experiments/300_epoch_modular_vpg1639499752.7267418"
experiment = "experiments/300_epoch_vpg1639510363.1020164"
experiment = "experiments/300_epoch_modular_ppo1639519677.8451397"
experiment = "experiments/100_epoch_ppo1639524434.8046937"
# experiment = "experiments/ppo_good"

if experiment is None:
    experiment = "experiments/" + sorted(os.listdir("experiments"))[-1]

df = pd.read_csv(f"{experiment}/progress.txt", sep="\t")


fig, axes = plt.subplots(nrows=4, ncols=1)

df.plot(x="Epoch", y= ["AverageEpRet", "MaxEpRet", "MinEpRet"], ax=axes[0]).legend(loc="lower left")
df.plot(x="Epoch", y="LossPi", ax=axes[1]).legend(loc="lower left")
df.plot(x="Epoch", y="LossV", ax=axes[2]).legend(loc="lower left")
df.plot(x="Epoch", y="Entropy", ax=axes[3]).legend(loc="lower left")
plt.show()


experiments = [
    # "experiments/ppo_good",
    "experiments/ppo_best",
    "experiments/ppo_modular_good",
    "experiments/vpg_baad",
    "experiments/vpg_modular_good",
]

data_frames = [pd.read_csv(f"{e}/progress.txt", sep="\t") for e in experiments]

for i, df in enumerate(data_frames):
    x = df["Epoch"][:100]
    y = df["AverageEpRet"][:100]
    label=experiments[i].split("/")[-1][:-5]
    plt.plot(x, y, label=label)

plt.ylabel("Average Episode Return")
plt.xlabel("Epochs")
plt.legend(loc="lower right")
plt.show()
