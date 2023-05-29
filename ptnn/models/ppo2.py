import random
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from ptnn.wrappers.ppo2Wrapper import Wrapper
from ptnn.layers.PositionalEncoding import PositionalEncoding
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class ActorCritic(nn.Module):
    def __init__(self, obs_size, max_seq_len, action_size):
        super().__init__()
        self.pe = PositionalEncoding(obs_size, max_seq_len)

        state_size = obs_size * max_seq_len
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
        state = state.view(state.size(0), -1)

        action_probs = self.actor(state)
        value = self.critic(state)

        return action_probs, value


def ppo2(env_name):
    # Set hyperparameters
    num_epochs = 500
    num_steps = 1024
    mini_batch_epochs = 10
    mini_batch_size = 128
    learning_rate = 1e-4
    gamma = 0.99
    clip_param = 0.2
    value_coeff = 0.5
    entropy_coeff = 0.01
    max_seq_len = 3
    exploration_rate = 0.0001
    noise_std_dev = 0.001

    # Create the environment
    env = gym.make(env_name, max_episode_steps=num_steps, render_mode="human")
    env = Wrapper(env, max_seq_len)

    # Initialize the model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActorCritic(
        env.observation_space.shape[0], max_seq_len, env.action_space.n).to(device)
    model.load_state_dict(torch.load('models/model.pth'))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter()

    # Training loop
    max_score = 0
    episode_durations = []
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

            if random.uniform(0, 1) < exploration_rate:
                action = torch.tensor(
                    [env.action_space.sample()], dtype=torch.long).to(device)
            else:
                action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(
                action.item())
            done = terminated or truncated
            log_prob = dist.log_prob(action)

            states.append(state.squeeze(dim=0))
            actions.append(action.squeeze(dim=0))
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob.squeeze(dim=0))

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
                advantages = advantages.to(device)

                # Add parameter noise
                for param in model.parameters():
                    noise = torch.randn_like(param) * noise_std_dev
                    param.data.add_(noise)

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
            writer.add_scalar(
                "Score/Average", torch.FloatTensor(episode_durations).mean().item(), epoch)
        writer.add_scalar("Loss/Policy", policy_loss.item(), epoch)
        writer.add_scalar("Loss/Value", value_loss.item(), epoch)
        writer.add_scalar("Loss/Entropy", entropy_loss.item(), epoch)
        writer.add_scalar("Loss/Total", total_loss.item(), epoch)

    torch.save(model.state_dict(), 'models/model.pth')
    writer.close()


def start():
    env_name = "CartPole-v1"
    ppo2(env_name)
