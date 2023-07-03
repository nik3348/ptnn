import math
import random
import gymnasium as gym
import matplotlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gymnasium.wrappers.human_rendering import HumanRenderingWrapper
from collections import namedtuple, deque
from itertools import count
from matplotlib import pyplot as plt

from ptnn.layers.Convolutional import ConvolutionalNeuralNetwork
from ptnn.layers.Encoder import Encoder
from ptnn.layers.PositionalEncoding import PositionalEncoding
from ptnn.wrappers.Wrapper import Wrapper

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Model(nn.Module):
    def __init__(self, embed_size, heads, output_size):
        super().__init__()
        self.cnn = ConvolutionalNeuralNetwork()
        self.posEmb = PositionalEncoding(embed_size)
        self.encoder = Encoder(embed_size, heads)
        self.linear = nn.Linear(embed_size, output_size)

    def forward(self, x):
        x = self.cnn(x)
        x = self.posEmb(x)
        x = self.encoder(x)
        logits = self.linear(x)
        return F.softmax(logits, dim=1)

class DQN():
    def __init__(self):
        self.env = gym.make("CartPole-v1", render_mode="human")
        self.env = HumanRenderingWrapper(self.env)
        self.env = Wrapper(self.env)
        self.env.reset()

        self.steps_done = 0
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # self. is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # LR is the learning rate of the ``AdamW`` optimizer
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.EMBED_SIZE = 16
        self.HEADS = 8
        self.NUM_CLASSES = self.env.action_space.n

    def train(self, args):
        policy_net = Model(self.EMBED_SIZE, self.HEADS, self.NUM_CLASSES).to(args.device)
        # policy_net.load_state_dict(torch.load('model.pth'))
        target_net = Model(self.EMBED_SIZE, self.HEADS, self.NUM_CLASSES).to(args.device)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=self.LR, amsgrad=True)
        memory = ReplayMemory(10000)

        def select_action(state):
            sample = random.random()
            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(-1. * self.steps_done / self.EPS_DECAY)
            self.steps_done += 1
            if sample > eps_threshold:
                with torch.no_grad():
                    return torch.argmax(policy_net(state.to('cuda')))
            else:
                return torch.tensor([self.env.action_space.sample()], dtype=torch.long)

        def optimize_model():
            if len(memory) < self.BATCH_SIZE:
                return
            transitions = memory.sample(self.BATCH_SIZE)
            batch = Transition(*zip(*transitions))

            for i in range(self.BATCH_SIZE):
                if batch.next_state[i] is None:
                    continue

                state_action_values = policy_net(batch.state[i].to('cuda'))

                with torch.no_grad():
                    next_state_values = target_net(batch.next_state[i].to('cuda'))

                expected_state_action_values = (next_state_values * self.GAMMA) + batch.reward[i].to('cuda')
                criterion = nn.SmoothL1Loss()

                loss = criterion(state_action_values, expected_state_action_values)

                optimizer.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
                optimizer.step()

        episode_durations = []
        def plot_durations(show_result=False):
            plt.figure(1)
            durations_t = torch.tensor(episode_durations, dtype=torch.float)
            if show_result:
                plt.title('Result')
            else:
                plt.clf()
                plt.title('Training...')

            plt.xlabel('Episode')
            plt.ylabel('Duration')
            plt.plot(durations_t.numpy())

            if len(durations_t) >= 100:
                means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
                means = torch.cat((torch.zeros(99), means))
                plt.plot(means.numpy())

            plt.pause(0.001)  # pause a bit so that plots are updated
            if is_ipython:
                if not show_result:
                    display.display(plt.gcf())
                    display.clear_output(wait=True)
                else:
                    display.display(plt.gcf())

        for _ in range(30):
            state, _ = self.env.reset()
            for t in count():
                action = select_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward])
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = observation

                memory.push(state, action, next_state, reward)
                state = next_state

                optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1-self.TAU)
                target_net.load_state_dict(target_net_state_dict)

                if done:
                    episode_durations.append(t + 1)
                    plot_durations()
                    break
        
        torch.save(policy_net.state_dict(), 'model.pth')
        print('Complete')
        plot_durations(show_result=True)
        plt.ioff()
        plt.show()
