import gymnasium
import numpy as np

class Wrapper(gymnasium.Wrapper):
    def __init__(self, env, max_seq_len=2):
        super().__init__(env)
        self.current_state = None
        self.d_model = self.env.observation_space.shape[0]
        self.max_seq_len = max_seq_len

    def step(self, action):
        new_state, reward, terminated, truncated, info = self.env.step(action)

        self.current_state = np.concatenate((self.current_state[1:], new_state.reshape(1, -1)), axis=0)
        return self.current_state, reward, terminated, truncated, info

    def reset(self):
        state, info = self.env.reset()

        state_size = self.d_model * self.max_seq_len
        input_array = np.random.randn(state_size)
        mask_array = np.zeros(state_size, dtype=np.float32)

        masked_input_array = np.where(mask_array == 0, 0, input_array)
        masked_input_array = masked_input_array.reshape((self.max_seq_len, self.d_model))

        self.current_state = np.concatenate((masked_input_array[1:], state.reshape(1, -1)), axis=0)
        return self.current_state, info
