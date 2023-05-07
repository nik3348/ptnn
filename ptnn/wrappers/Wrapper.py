import gymnasium as gym
import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import crop

class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        obs = self.get_state()
        return obs, reward, terminated, truncated, info

    def reset(self):
        _, info = self.env.reset()
        return self.get_state(), info

    def get_state(self):
        image_tensor = self.env.render()
        image_tensor = torch.div(torch.from_numpy(image_tensor).unsqueeze(0), 255).float()

        target_size = (128, 128)
        resize_transform = Resize(target_size)
        resized_tensor = resize_transform(image_tensor.permute(0, 3, 1, 2))
        
        return resized_tensor
