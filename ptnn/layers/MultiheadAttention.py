import torch
import torch.nn as nn

class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        self.qlinear = nn.Linear(embed_size, embed_size)
        self.klinear = nn.Linear(embed_size, embed_size)
        self.vlinear = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=1)
        self.linear = nn.Linear(embed_size, embed_size)

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

    def forward(self, query, keys, values):
        query = self.qlinear(query)
        query = query.view(self.heads, self.head_dim)
        
        keys = self.klinear(keys)
        keys = keys.view(self.heads, self.head_dim)

        values = self.vlinear(values)
        values = values.view(self.heads, self.head_dim)

        x = torch.matmul(query, torch.t(keys))
        x /= self.head_dim ** (1/2)
        # Masking optional here
        x = self.softmax(x)
        x = torch.matmul(x, values)
        x = x.view(self.embed_size)
        x = self.linear(x)
        return x
