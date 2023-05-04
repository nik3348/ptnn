import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.layers(x)
        return x
