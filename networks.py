from math import log
from gym.spaces.box import Box
import joblib
import numpy as np
import torch
from torch._C import Value
from torch.distributions import transforms
from torch.distributions.normal import Normal
import torch.nn as nn
from torch.optim.adam import Adam
from torchvision.models import efficientnet_b0
from torch.distributions.categorical import Categorical


class StateEncoder(nn.Module):
    def __init__(self, state_dimensions, *args) -> None:
        super().__init__()
        self.width, self.height, self.channels = state_dimensions

        self.encoder = nn.Sequential(
            nn.Conv2d(self.channels, 32, kernel_size=8, stride=2),
            # nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            # nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.ReLU(),
            # nn.Flatten()
        )

    def forward(self, state: torch.Tensor, *args):
        x = self.encoder(state)
        return x.view(x.size(0), -1)
    
    def conv_forward(self, state):
        return self.encoder(state)

    def set_grad(self, require_grad):
        for p in self.encoder.parameters():
            p.requires_grad = require_grad

    @property
    def latent_size(self):
        height = self.height
        width = self.width
        channels = self.channels
        print(height, width, channels)
        def conv_size(in_size, kernel_size, stride):
            return (in_size - kernel_size) // stride + 1

        for layer in self.encoder.modules():
            if isinstance(layer, nn.Conv2d):
                layer: nn.Conv2d  # Type hint for simplicity
                kh, kw = layer.kernel_size
                sh, sw = layer.stride
                height = conv_size(height, kh, sh)
                width = conv_size(width, kh, sh)
                channels = layer.out_channels
            # TODO add logic for other size manipulating layers
        return channels * height * width, (channels, height, width)

class StateDecoder(nn.Module):
    def __init__(self, observation_dims, latent_conv_shape, *args) -> None:
        super().__init__()

        self.latent_conv_shape = latent_conv_shape
        self.width, self.height, self.channels = observation_dims

        channels_in = latent_conv_shape[0]

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels_in, 64, kernel_size=3, stride=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, output_padding=1),
            # nn.ReLU(),
            nn.ConvTranspose2d(32, self.channels, kernel_size=8, stride=2),
            nn.Sigmoid()
        )

    def forward(self, encoded_state: torch.Tensor, *args):
        decoded = self.decoder(encoded_state)

        # Change shape from (channel, height, width) to (height, width, channel) 
        # decoded = decoded.permute(1, 2, 0)
        return decoded


class PolicyHead(nn.Module):
    def __init__(self, action_space, latent_size, *args) -> None:
        super().__init__()

        action_shape = action_space.shape[0]

        self.policy_head = nn.Sequential(
            nn.Linear(latent_size, 100),
            nn.ReLU(),
            nn.Linear(100, action_shape)
        )
        log_std = -0.5 * np.ones(action_shape, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))


    def forward(self, state):
        logits = self.policy_head(state)
        std = torch.exp(self.log_std)
        return Normal(logits, std)

class CategoricalPolicyHead(nn.Module):
    def __init__(self, num_actions, latent_size, *args) -> None:
        super().__init__()

        self.policy_head = nn.Sequential(
            nn.Linear(latent_size, 100),
            nn.ReLU(),
            nn.Linear(100, num_actions)
        )

    def forward(self, state):
        logits = self.policy_head(state)
        return Categorical(logits=logits)

class ValueHead(nn.Module):
    def __init__(self, latent_size, *args) -> None:
        super().__init__()

        self.value_head = nn.Sequential(
            nn.Linear(latent_size, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )

    def forward(self, state):
        return self.value_head(state)

class CatecoricalAC(nn.Module):
    def __init__(self, state_dimensions, num_actions, **kwargs) -> None:
        super().__init__()

        self.encoder = StateEncoder(state_dimensions)
        self.action_list = num_actions
        latent_size, _ = self.encoder.latent_size
        self.policy_head = CategoricalPolicyHead(num_actions, latent_size)
        self.value_head = ValueHead(latent_size)
        if "model_state_path" in kwargs:
            with open(kwargs["model_state_path"], "rb") as f:
                self.load_state_dict(joblib.load(f))

    def _forward_pass(self, state):
        if len(state.shape) != 4:
            state = state[None]  # Add dummy dimension
        encoded_state = self.encoder(state)
        policy = self.policy_head(encoded_state)
        v = self.value_head(encoded_state)
        return policy, v

    @property
    def pi(self):
        return nn.Sequential(self.encoder, self.policy_head)
    
    @property
    def v(self):
        return nn.Sequential(self.encoder, self.value_head)

    def no_grad_forward(self, state):
        with torch.no_grad():
            encoded_state = self.encoder(state)
            policy = self.policy_head(encoded_state)
            v = self.value_head(encoded_state)
        return policy, v

    def no_grad_step(self, state):
        with torch.no_grad():
            pi, v = self.no_grad_forward(state)
            action = pi.sample().squeeze()
            logp = pi.log_prob(action)
        return action.numpy(), v.numpy(), logp.numpy()

    def step(self, state):
        with torch.no_grad():
            pi, v = self._forward_pass(state)
            action = pi.sample().squeeze()
            logp = pi.log_prob(action)
        return action.numpy(), v.numpy(), logp.numpy()

    def forward(self, state, action=None):
        pi, v = self._forward_pass(state)
        logp = None
        if action is not None:
            logp = pi.log_prob(action.squeeze())
        return pi, v, logp


class ModularCatecoricalAC(nn.Module):
    def __init__(self, observation_space, action_space, **kwargs) -> None:
        super().__init__()

        self.encoder = StateEncoder(observation_space)

        latent_size, latent_conv_shape = self.encoder.latent_size
        self.decoder = StateDecoder(observation_space, latent_conv_shape)
        self.policy_head = CategoricalPolicyHead(action_space, latent_size)
        self.value_head = ValueHead(latent_size)

        if "model_state_path" in kwargs:
            with open(kwargs["model_state_path"], "rb") as f:
                self.load_state_dict(joblib.load(f))

        if "encoder_path" in kwargs:
            with open(kwargs["encoder_path"], "rb") as f:
                self.encoder.load_state_dict(torch.load(f))

    def parameters(self):
        return list(self.policy_head.parameters()) + list(self.value_head.parameters())

    @property
    def pi(self):
        return self.policy_head
    
    @property
    def v(self):
        return self.value_head

    def autoencoder_parameters(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def _forward_pass(self, state):
        with torch.no_grad():
            encoded_state = self.encoder(state)
        policy = self.policy_head(encoded_state)
        v = self.value_head(encoded_state)
        return policy, v

    def step(self, state):
        with torch.no_grad():
            pi, v = self._forward_pass(state)
            action = pi.sample().squeeze()
            logp = pi.log_prob(action)
        return action.numpy(), v.numpy(), logp.numpy()

    def forward(self, state, action=None):
        pi, v = self._forward_pass(state)
        logp = None
        if action is not None:
            logp = logp = pi.log_prob(action.squeeze())
        return pi, v, logp

    def forward_autoenc(self, state, grad=True):
        with torch.set_grad_enabled(grad):
            encoded = self.encoder.conv_forward(state)
            out = self.decoder(encoded)
        return out

def compare_params(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        res = p1.data == p2.data
        print(res.all())

    



if __name__ == "__main__":
    from environment import MyEnv
    from pprint import pprint

    env = MyEnv()
    ac = ModularCatecoricalAC(env.observation_space, env.action_space)
    ref_encoder = StateEncoder(env.observation_space)
    ref_encoder.load_state_dict(ac.encoder.state_dict())
    ref_pol = CategoricalPolicyHead(env.action_space, ac.encoder.latent_size[0])
    ref_pol.load_state_dict(ac.policy_head.state_dict())
    ref_val = ValueHead(ac.encoder.latent_size[0])
    ref_val.load_state_dict(ac.value_head.state_dict())
    print("encoder first time")
    compare_params(ref_encoder, ac.encoder)
    print("policy first time")
    compare_params(ref_pol, ac.policy_head)
    print("value first time")
    compare_params(ref_val, ac.value_head)
    ac2 = ModularCatecoricalAC(env.observation_space, env.action_space, envoder_path="tmp.pt")

    torch.save(ac.state_dict(), "tmp.pt")
    ac2.load_state_dict(torch.load(open("tmp.pt", "rb")))

    print("encoder second time")
    compare_params(ac2, ac)
    # print("policy second time")
    # compare_params(ref_pol, ac.policy_head)
    # print("value second time")
    # compare_params(ref_val, ac.value_head)


