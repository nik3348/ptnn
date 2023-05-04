import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_size):
        super().__init__()
        self.pos_enc = torch.zeros(max_seq_len, embed_size)

        for pos in range(max_seq_len):
            for i in range(0, embed_size, 2):
                self.pos_enc[pos, i] = math.sin(pos / (10000 ** (i / embed_size)))
                self.pos_enc[pos, i+1] = math.cos(pos / (10000 ** (i / embed_size)))

    def forward(self, x):
        x = x.clone().detach()
        return x + self.pos_enc
