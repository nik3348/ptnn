import gymnasium as gym
import numpy as np

class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_state = None

    def step(self, action):
        new_state, reward, terminated, truncated, info = self.env.step(action)
        self.current_state = np.concatenate((self.current_state[1:], new_state.reshape(1, -1)), axis=0)
        return self.current_state, reward, terminated, truncated, info

    def reset(self):
        state, info = self.env.reset()

        input_array = np.random.randn(8)
        mask_array = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        masked_input_array = np.where(mask_array == 0, -1e9, input_array)
        masked_input_array = masked_input_array.reshape((2, 4))

        self.current_state = np.vstack((masked_input_array, state))
        return self.current_state, info
