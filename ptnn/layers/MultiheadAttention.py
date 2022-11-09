import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(embed_size, embed_size)

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

    def forward(self, query, keys, values):
        query = query.view(self.heads, self.head_dim)
        keys = keys.view(self.heads, self.head_dim)
        values = values.view(self.heads, self.head_dim)

        x = torch.matmul(query, torch.t(keys))
        x /= self.head_dim ** (1/2)
        x = self.softmax(x)
        x = torch.matmul(x, values)
        x = x.view(self.embed_size)
        x = self.linear(x)
        return x
