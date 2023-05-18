from itertools import count
import gymnasium as gym
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from ptnn.wrappers.ppo2Wrapper import Wrapper
from ptnn.layers.PositionalEncoding import PositionalEncoding
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.pe = PositionalEncoding(4, 3)
        self.actor = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        state = self.pe(state)

        action_probs = self.actor(state)
        value = self.critic(state)

        return action_probs, value

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

def ppo2(env_name):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the environment
    env = gym.make(env_name, render_mode="human")
    env = Wrapper(env)

    # Set hyperparameters
    num_epochs = 500
    num_steps = 2048
    mini_batch_epochs = 5
    mini_batch_size = 64
    learning_rate = 1e-4
    gamma = 0.99
    clip_param = 0.2
    value_coeff = 0.5
    entropy_coeff = 0.01

    # Initialize the model and optimizer
    state_size = env.observation_space.shape[0] * 3
    action_size = env.action_space.n
    model = ActorCritic(state_size, action_size).to(device)
    model.load_state_dict(torch.load('models/model.pth'))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter()

    # Training loop
    max_score = 0
    for epoch in range(num_epochs):
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []

        state, _ = env.reset()
        for score in range(num_steps):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            with torch.no_grad():
                action_probs, value = model(state)
            dist = Categorical(logits=action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)

            state = next_state

            if done:
                episode_durations.append(score)
                if score > max_score:
                    max_score = score
                    writer.add_scalar("Score/Max", max_score, epoch)
                break

        # Compute returns and advantages
        returns = []
        advantages = []
        R = 0
        for t in reversed(range(len(rewards))):
            R = rewards[t] + gamma * R
            returns.insert(0, R)
            advantages.insert(0, R - values[t])

        # Normalize advantages
        advantages = torch.FloatTensor(advantages).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update policy
        dataset = list(zip(states, actions, log_probs, returns, advantages))
        data_loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)

        for _ in range(mini_batch_epochs):
            for states, actions, old_log_probs, returns, advantages in data_loader:
                states = states.to(device)
                actions = actions.to(device)
                old_log_probs = old_log_probs.to(device)
                returns = returns.to(device)
                advantages = advantages.unsqueeze(1).to(device)

                new_action_probs, value_pred = model(states)
                dist = Categorical(logits=new_action_probs)
                new_log_probs = dist.log_prob(actions)
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surrogate1 = ratio * advantages
                surrogate2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages

                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_loss = F.mse_loss(value_pred.squeeze().float(), returns.float())

                entropy_loss = dist.entropy().mean()

                total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        if epoch > 100:
            writer.add_scalar("Score/Average", torch.FloatTensor(episode_durations).mean().item(), epoch)
        writer.add_scalar("Loss/Policy", policy_loss.item(), epoch)
        writer.add_scalar("Loss/Value", value_loss.item(), epoch)
        writer.add_scalar("Loss/Entropy", entropy_loss.item(), epoch)
        writer.add_scalar("Loss/Total", total_loss.item(), epoch)

    torch.save(model.state_dict(), 'models/model.pth')
    writer.close()

def start():
    env_name = "CartPole-v1"
    ppo2(env_name)
