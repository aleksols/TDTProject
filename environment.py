import gym
from numpy.core.shape_base import stack
import torch
import torchvision.transforms as T
import numpy as np

class MyEnv():

    def __init__(self, stack_images=4, skip_actions=4, grayscale=True, gas_factor=1, break_factor=1) -> None:
        self.env = gym.make("CarRacing-v0", verbose=0)
        self.stack_images = stack_images
        self.skip_actions = skip_actions
        self.grayscale = grayscale
        self.action_list = np.array([[0, 0, 0], [-1, 0, 0], [1, 0, 0], [0, 1 * gas_factor, 0],  [0, 0, 1 * break_factor]])
        self.observation_space = (66, 66, stack_images)

        self.transform_state = T.Compose([T.ToTensor(), T.ToPILImage(), T.Grayscale(), T.ToTensor()])
        if not grayscale:
            self.transform_state = T.Compose([T.ToTensor(), T.ToPILImage(), T.ToTensor()])
            self.observation_space = (66, 66, stack_images*3)
        self.action_space = self.action_list.shape[0]

    def _get_tensor_state(self, state):
        state = state.copy()
        # cut track features from image
        state = self.transform_state(state)
        state = state[None, :, :66, 15:81] # Add dummy dimension here
        return state

    def _apply_action(self, action):
        if action is not None:
            action = self.action_list[action]  # Actions are indices
        accumulated_reward = 0
        states = []
        d = False
        for i in range(self.skip_actions):
            s, r, d, _ = self.env.step(action)
            accumulated_reward += r
            s = self._get_tensor_state(s)
            states.append(s)
        state = torch.cat(states[-self.stack_images:] ,dim=1)

        return state, accumulated_reward, d


    def reset(self):
        s = self.env.reset()
        states = []
        for _ in range(50):
            s, _, _, _ = self.env.step(None)
            states.append(s)
        states = states[-self.stack_images:]
        
        return torch.cat([self._get_tensor_state(s) for s in states], dim=1)

    def step(self, action):
        return self._apply_action(action)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
    


if __name__ == "__main__":
    e = MyEnv()
    s = e.reset()
    print(s.shape)