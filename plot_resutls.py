import matplotlib.pyplot as plt
import pandas as pd
import os

# experiment = "experiments/1639296268.9138777"
experiment = "experiments/100_epoch_ppo1639398377.608867"
experiment = "experiments/300_epoch_modular_vpg1639479473.031819"

if experiment is None:
    experiment = "experiments/" + sorted(os.listdir("experiments"))[-1]

df = pd.read_csv(f"{experiment}/progress.txt", sep="\t")


fig, axes = plt.subplots(nrows=4, ncols=1)

df.plot(x="Epoch", y= ["AverageEpRet", "MaxEpRet", "MinEpRet"], ax=axes[0]).legend(loc="lower left")
df.plot(x="Epoch", y="LossPi", ax=axes[1]).legend(loc="lower left")
df.plot(x="Epoch", y="LossV", ax=axes[2]).legend(loc="lower left")
df.plot(x="Epoch", y="Entropy", ax=axes[3]).legend(loc="lower left")
plt.show()