import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiheadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiheadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * heads ==
                embed_size), "Embed size needs to be div by heads"

        self.qlinear = nn.Linear(embed_size, embed_size)
        self.klinear = nn.Linear(embed_size, embed_size)
        self.vlinear = nn.Linear(embed_size, embed_size)

        self.q_ln = nn.LayerNorm(embed_size)
        self.k_ln = nn.LayerNorm(embed_size)
        self.v_ln = nn.LayerNorm(embed_size)

        self.dropout = nn.Dropout(0.1)

        self.linear = nn.Linear(embed_size, embed_size)

    def forward(self, query, keys, values):
        query = self.qlinear(query)
        query = self.q_ln(query)
        query = query.view(-1, self.heads, self.head_dim)

        keys = self.klinear(keys)
        keys = self.k_ln(keys)
        keys = keys.view(-1, self.heads, self.head_dim)

        values = self.vlinear(values)
        values = self.v_ln(values)
        values = values.view(-1, self.heads, self.head_dim)

        x = torch.matmul(query, keys.transpose(1, 2))
        x /= self.head_dim ** (1/2)
        # Masking optional here
        x = F.softmax(x, dim=-1)

        x = self.dropout(x)

        x = torch.matmul(x, values)
        x = x.view(-1, self.embed_size)
        x = self.linear(x)

        return x
