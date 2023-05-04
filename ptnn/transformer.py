import argparse
import torch
import torch.nn as nn
from ptnn.layers.ConvolutionalNeuralNetwork import ConvolutionalNeuralNetwork
from ptnn.layers.Encoder import Encoder
from ptnn.layers.FeedForward import FeedForward
from ptnn.layers.PositionalEmbedding import PositionalEmbedding
from ptnn.layers.MultiheadAttention import MultiheadAttention

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
print("GPU enabled:", torch.cuda.is_available())
if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

EMBED_SIZE = 512
HEADS = 8

if __name__ == '__main__':
    cnn = ConvolutionalNeuralNetwork()
    x = torch.randn(1, 3, 32, 32)
    x = cnn.forward(x)

    posEmb = PositionalEmbedding(1, EMBED_SIZE)
    x = posEmb.forward(torch.tensor(x))

    model = Encoder(EMBED_SIZE, HEADS)
    x = model.forward(x)
    print(x.size())
