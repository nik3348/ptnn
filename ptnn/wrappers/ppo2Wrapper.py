import gymnasium as gym
import numpy as np

class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.current_state = None

    def step(self, action):
        new_state, reward, terminated, truncated, info = self.env.step(action)
        
        cur_state = np.array(self.current_state[4:])
        concatenated_array = np.concatenate((cur_state, new_state), axis=0)
        self.current_state = concatenated_array
        return concatenated_array, reward, terminated, truncated, info

    def reset(self):
        state, info = self.env.reset()

        input_array = np.random.randn(4)
        mask_array = np.array([0, 0, 0, 0], dtype=np.float32)
        masked_input_array = np.where(mask_array == 0, -1e9, input_array)

        concatenated_array = np.concatenate((masked_input_array, state), axis=0)
        self.current_state = concatenated_array
        return concatenated_array, info
