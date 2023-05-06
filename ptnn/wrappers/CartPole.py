import numpy as np
import gymnasium as gym

from gymnasium.spaces import Box
from ray.rllib.utils.images import resize

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, dim):
        gym.ObservationWrapper.__init__(self, env)
        self.width = dim
        self.height = dim
        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
        )

    def observation(self, frame):
        frame = resize(frame, height=self.height, width=self.width)
        return frame[:, :, None]
