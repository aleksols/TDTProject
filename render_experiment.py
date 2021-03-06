import matplotlib.pyplot as plt
import pandas as pd
import os
from environment import MyEnv
import json
import torch
from networks import CatecoricalAC

# experiment = "experiments/1639296268.9138777"
experiment = "experiments/100_epoch_modular_ppo"
experiment = "experiments/100_epoch_vpg1639496268.0265865"
experiment = "experiments/100_epoch_ppo1639491161.8668485"
experiment = "experiments/100_epoch_ppo1639524434.8046937"
# experiment = "experiments/300_epoch_modular_vpg1639499752.7267418"

model = 20

if experiment is None:
    experiment = "experiments/" + sorted(os.listdir("experiments"))[-1]

config_path = experiment + "/config.json"
model_path = experiment + f"/pyt_save/model{model}.pt"

with open(config_path, "r") as f:
    config = json.load(f)

env = MyEnv(stack_images=config["image_stack"], skip_actions=config["action_skip"], grayscale=config["grayscale_transform"], gas_factor=config["gas_factor"], break_factor=config["break_factor"])
# env = MyEnv(
#     stack_images=config["image_stack"], 
#     skip_actions=config["action_skip"], 
#     grayscale=config["grayscale_transform"], 
#     gas_factor=0.3, break_factor=1)


ac = torch.load(model_path)

state = env.reset()

done = False
cum_rew = 0
while not done:
    a, _, _ = ac.step(state)
    state, r, done = env.step(a)
    env.render()

    cum_rew += r
print(cum_rew)