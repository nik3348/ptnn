import torch.nn as nn
from ptnn.layers.MultiheadAttention import MultiheadAttention 
from ptnn.layers.FeedForward import FeedForward 


class Encoder(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.mha = MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        base = x
        x = self.mha(x, x, x)
        base += self.norm1(x)
        x = base
        x = self.ff(x)
        x = base + self.norm2(x)
        return x
