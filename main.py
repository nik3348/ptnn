import argparse
import math
import random
import gymnasium as gym
import matplotlib
import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple, deque
from itertools import count
from matplotlib import pyplot as plt

from ptnn.Model import Model
from ptnn.env.Wrapper import Wrapper

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

env = gym.make("CartPole-v1", render_mode="human")
env = Wrapper(env)
env.reset()

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
EMBED_SIZE = 512
HEADS = 8
NUM_CLASSES = env.action_space.n

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


if __name__ == '__main__':
    policy_net = Model(EMBED_SIZE, HEADS, NUM_CLASSES).to(args.device)
    policy_net.load_state_dict(torch.load('model.pth'))
    target_net = Model(EMBED_SIZE, HEADS, NUM_CLASSES).to(args.device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    steps_done = 0

    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state.to('cuda')).max(1)[1]
        else:
            return torch.tensor([env.action_space.sample()], dtype=torch.long)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return
        transitions = memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        for i in range(BATCH_SIZE):
            if batch.next_state[i] is None:
                continue

            state_action_values = policy_net(batch.state[i].to('cuda')).gather(1, batch.action[i].to('cuda').unsqueeze(1))

            with torch.no_grad():
                next_state_values = target_net(batch.next_state[i].to('cuda')).max(1)[0]

            expected_state_action_values = (next_state_values * GAMMA) + batch.reward[i].to('cuda')
            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

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
        state, _ = env.reset()
        for t in count():
            action = select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
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
                target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
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
