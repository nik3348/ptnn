import torch.nn as nn
from ptnn.layers.MultiheadAttention import MultiheadAttention 
from ptnn.layers.FeedForward import FeedForward 

class Encoder(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(Encoder, self).__init__()
        self.mha = MultiheadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = FeedForward(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.mha(x, x, x)
        x = self.dropout(residual + x)

        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(residual + x)
        return x
