from itertools import count
import math
import random
import gymnasium as gym
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ptnn.layers.Convolutional import Convolutional
from ptnn.wrappers.Wrapper import Wrapper

from ncps.torch import CfC
from ncps.wirings import AutoNCP

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
    
def start():
    env = gym.make("CartPole-v1", render_mode="human")
    env = Wrapper(env)
    observation, _ = env.reset()

    hx = None
    num_episodes = []
    memory = []
    gamma = 0.99
    lr = 1e-4
    tau = 0.005

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_block = Convolutional()
            wiring = AutoNCP(16, 2)
            input_size = 32
            self.rnn = CfC(input_size, wiring)
        
        def forward(self, x, hx):
            x = self.conv_block(x)
            return self.rnn(x, hx)

    model = Model().to("cuda")
    # model.load_state_dict(torch.load('model.pth'))
    target = Model().to("cuda")
    target.load_state_dict(model.state_dict())
    steps = 0
    optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True)

    def plot_durations(show_result=False):
            plt.figure(1)
            durations_t = torch.tensor(num_episodes, dtype=torch.float)
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

    for _ in range(100):
        for t in count():
            steps += 1
            with torch.no_grad():
                pred, hx = model(observation.to('cuda'), hx)
            action = pred.squeeze(0).squeeze(0).argmax().item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward = t
            if done:
                reward = -1
            memory.append((observation, action, reward, next_state, done))
            observation = next_state

            if len(memory) > 20:
                batch = random.sample(memory, 20)
                state_batch = torch.cat([b[0] for b in batch], dim=0)
                action_batch = torch.tensor([b[1] for b in batch], dtype=torch.long)
                reward_batch = torch.tensor([b[2] for b in batch], dtype=torch.float)
                next_state_batch = torch.cat([b[3] for b in batch], dim=0)

                state_action_values = model(state_batch.to('cuda'), hx)[0].gather(1, action_batch.to('cuda').unsqueeze(1))
                with torch.no_grad():
                    next_state_values = target(next_state_batch.to('cuda'), hx)[0].max(1)[0].detach()

                expected_state_action_values = -reward_batch.to('cuda') + gamma * next_state_values
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_value_(model.parameters(), 100)
                optimizer.step()

            if len(memory) % 2 == 0:
                target.load_state_dict(model.state_dict())    

            target_net_state_dict = model.state_dict()
            policy_net_state_dict = target.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1-tau)
            target.load_state_dict(target_net_state_dict)

            if done:
                plot_durations()
                observation, _ = env.reset()
                hx = None
                num_episodes.append(t + 1)

                break

    torch.save(model.state_dict(), 'model.pth')
    plot_durations(True)
    ob = observation
    arr_ = np.squeeze(ob)
    plt.imshow(arr_.permute(1, 2, 0))
    plt.show()
