import argparse
import torch
import torch.nn as nn
from ptnn.layers.MultiheadAttention import MultiheadAttention 
from ptnn.layers.FeedForward import FeedForward 

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

def start():
    print('hello world')
    model = MultiheadAttention(EMBED_SIZE, HEADS)
    q = torch.rand(EMBED_SIZE)
    k = torch.rand(EMBED_SIZE)
    v = torch.rand(EMBED_SIZE)
    x = model.forward(q, k, v)
    print(x.size())
