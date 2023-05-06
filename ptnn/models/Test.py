import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from ncps.datasets.torch import AtariCloningDataset
from ray.rllib.env.wrappers.atari_wrappers import wrap_deepmind
from tqdm import tqdm
from ptnn.layers.ConvBlock import ConvBlock

from ptnn.layers.ConvCfc import ConvCfC
from ptnn.layers.Convolutional import Convolutional
from ptnn.wrappers.CartPole import WarpFrame
from ptnn.wrappers.Wrapper import Wrapper
from gymnasium.wrappers.human_rendering import HumanRendering

from torchvision.transforms import Resize
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP

def start():
    env = gym.make("CartPole-v1", render_mode="human")
    env = Wrapper(env)
    # env = HumanRendering(env)
    observation, _ = env.reset()

    conv_block = Convolutional()
    wiring = AutoNCP(16, 2)
    input_size = 16
    rnn = CfC(input_size, wiring)
    hx = None
    returns = []
    total_reward = 0
    num_episodes = None

    for _ in range(10):
        while True:
            x = conv_block(observation)
            pred, hx = rnn(x, hx)
            action = pred.squeeze(0).squeeze(0).argmax().item()
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                # observation, _ = env.reset()
                hx = None
                returns.append(total_reward)
                total_reward = 0
                if num_episodes is not None:
                    num_episodes = num_episodes - 1
                    if num_episodes == 0:
                        return returns
                break

    ob = observation
    arr_ = np.squeeze(ob)
    plt.imshow(arr_.permute(1, 2, 0))
    plt.show()
