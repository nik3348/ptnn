import torch
import torch.nn as nn
from ptnn.layers.MultiheadAttention import MultiheadAttention 
from ptnn.layers.FeedForward import FeedForward 

class Encoder(nn.Module):
    def __init__(self, embed_size, heads):
        super(Encoder, self).__init__()
        self.mha = MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm()
        self.ff = FeedForward()

    def forward(self, x):

        return x
