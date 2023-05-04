import torch.nn as nn
import torch.nn.functional as F
from ptnn.layers.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from ptnn.layers.Encoder import Encoder
from ptnn.layers.PositionalEmbedding import PositionalEmbedding


class Model(nn.Module):
    def __init__(self, embed_size, heads, output_size):
        super().__init__()
        self.cnn = ConvolutionalNeuralNetwork()
        self.posEmb = PositionalEmbedding(1, embed_size)
        self.encoder = Encoder(embed_size, heads)
        self.linear = nn.Linear(embed_size, output_size)

    def forward(self, x):
        x = self.cnn(x)
        x = self.posEmb(x)
        x = self.encoder(x)
        logits = self.linear(x)
        return F.softmax(logits, dim=1)
