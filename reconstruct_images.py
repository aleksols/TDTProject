import torch
from networks import ModularCatecoricalAC
from environment import MyEnv
import matplotlib.pyplot as plt

exp_directory = "experiments/encoder_pretrain_no_relu"

env = MyEnv()

s = env.reset()

ac = ModularCatecoricalAC(env.observation_space, env.action_space)
with open(f"{exp_directory}/encoder_weights.pt", "rb") as f:
    ac.encoder.load_state_dict(torch.load(f))

with open(f"{exp_directory}/decoder_weights.pt", "rb") as f:
    ac.decoder.load_state_dict(torch.load(f))

step = 0

states = []
done = False
while not done:
    a, v, p = ac.step(s)
    s, _, _ = env.step(a)
    if step in [10, 50, 100]:
        states.append(s)

    step += 1
    if step > 100:
        break

env.close()

tensor_states = torch.cat(states)

reconstructed = ac.forward_autoenc(tensor_states, grad=False)



for img in range(3):
    pil_image = reconstructed[img, -1]
    orig = tensor_states[img, -1]
    fig, axes = plt.subplots(ncols=2, nrows=1)
    axes[0].imshow(orig, cmap="gray")
    axes[1].imshow(pil_image, cmap="gray")
    plt.show()

