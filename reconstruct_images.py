import torch
from networks import ModularCatecoricalAC
from environment import MyEnv
import matplotlib.pyplot as plt
import time

exp_directory = "experiments/encoder_pretrain_no_relu_66_img"

env = MyEnv()

s = env.reset()

ac = ModularCatecoricalAC(env.observation_space, env.action_space)
with open(f"{exp_directory}/encoder_weights.pt", "rb") as f:
    ac.encoder.load_state_dict(torch.load(f))

with open(f"{exp_directory}/decoder_weights.pt", "rb") as f:
    ac.decoder.load_state_dict(torch.load(f))


ac_path = "experiments/vpg_good/pyt_save/model0.pt"
with open("experiments/ppo_modular_good/pyt_save/model100.pt", "rb") as f:
    ac2 = torch.load(f)
step = 0

states = []
done = False
while not done:
    a, v, p = ac2.step(s)
    s, r, d = env.step(a)
    if d:
        s = env.reset()
    if step % 30 == 0:
        states.append(s)
    step += 1
    if step > 150:
        break

env.close()

tensor_states = torch.cat(states)

reconstructed = ac.forward_autoenc(tensor_states, grad=False)



for img in range(len(states)):
    pil_image = reconstructed[img, -1]
    orig = tensor_states[img, -1]
    fig, axes = plt.subplots(ncols=2, nrows=1)
    axes[0].imshow(orig, cmap="gray")
    axes[1].imshow(pil_image, cmap="gray")
    plt.show()

