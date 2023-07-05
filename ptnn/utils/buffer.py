import torch
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.tensor(state).float().to(device),
            torch.tensor(action).float().to(device),
            torch.tensor(reward).float().unsqueeze(1).to(device),
            torch.tensor(next_state).float().to(device),
            torch.tensor(done).float().unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)
