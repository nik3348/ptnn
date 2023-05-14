import argparse
import torch
from ptnn.models.ppo2 import start

parser = argparse.ArgumentParser(description='PyTorch Example')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
args = parser.parse_args()
args.device = None
if not args.disable_cuda and torch.cuda.is_available():
    # print("GPU enabled:", torch.cuda.is_available())
    args.device = torch.device('cuda')
else:
    args.device = torch.device('cpu')

if __name__ == '__main__':
    start()
