import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.l1 = nn.Linear(embed_size, embed_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return x