import gymnasium
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.distributions import Normal
from torch.utils.data import DataLoader
from supersuit import color_reduction_v0, frame_stack_v1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)

        # Initialize layer weights
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.mean.weight)
        init.xavier_uniform_(self.log_std.weight)

        # Initialize biases
        init.constant_(self.fc1.bias, 0.0)
        init.constant_(self.fc2.bias, 0.0)
        init.constant_(self.mean.bias, 0.0)
        init.constant_(self.log_std.bias, 0.0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        std = log_std.exp()
        return mean, std


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # Initialize layer weights
        init.xavier_uniform_(self.fc1.weight)
        init.xavier_uniform_(self.fc2.weight)
        init.xavier_uniform_(self.fc3.weight)

        # Initialize biases
        init.constant_(self.fc1.bias, 0.0)
        init.constant_(self.fc2.bias, 0.0)
        init.constant_(self.fc3.bias, 0.0)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_size=256, alpha=0.1, gamma=0.9999, tau=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_size).to(device)
        self.critic1 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.target_critic1 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_size).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=3e-4)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=3e-4)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, std = self.actor(state)
        dist = Normal(mean, std)
        action = dist.sample()
        return action.squeeze(0).detach().cpu().numpy()

    def update(self, state, action, next_state, reward, done):
        for _ in range(10):
            # Compute target Q value
            with torch.no_grad():
                next_action, next_log_prob = self.actor(next_state)
                target_q1 = self.target_critic1(next_state, next_action)
                target_q2 = self.target_critic2(next_state, next_action)
                target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
                target_q = reward + (1 - done.int()) * self.gamma * target_q

            # Update Critic networks
            current_q1 = self.critic1(state, action)
            current_q2 = self.critic2(state, action)
            critic1_loss = F.mse_loss(current_q1.float(), target_q.float())
            critic2_loss = F.mse_loss(current_q2.float(), target_q.float())

            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()

            # Update Actor network
            new_action, new_log_prob = self.actor(state)
            q1 = self.critic1(state, new_action)
            q2 = self.critic2(state, new_action)
            q = torch.min(q1, q2)

            actor_loss = (self.alpha * new_log_prob - q).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
            self.actor_optimizer.step()

            # Update target networks
            for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
            for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def save(self, directory):
        torch.save(self.actor.state_dict(), directory + "/actor.pth")
        torch.save(self.critic1.state_dict(), directory + "/critic1.pth")
        torch.save(self.critic2.state_dict(), directory + "/critic2.pth")

    def load(self, directory):
        self.actor.load_state_dict(torch.load(directory + "/actor.pth"))
        self.critic1.load_state_dict(torch.load(directory + "/critic1.pth"))
        self.critic2.load_state_dict(torch.load(directory + "/critic2.pth"))
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

def start():
    steps = 1000
    env = gymnasium.make('MountainCarContinuous-v0', max_episode_steps=steps, render_mode="human")
    env = frame_stack_v1(env, 4)

    model = SACAgent(env.observation_space.shape[0], env.action_space.shape[0])
    model.load('models/sac')
    data_loader = DataLoader(torch.load('data/offline.pth'), batch_size=128, shuffle=True)

    for _ in range(100):
        state, _ = env.reset()
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        for _ in range(steps):
            action = model.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = next_state.reshape((1, -1)).squeeze()
            
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

            state = next_state
            if done:
                break
        
        states = torch.FloatTensor(np.stack(states)).to(device)
        actions = torch.FloatTensor(np.stack(actions)).to(device)
        next_states = torch.FloatTensor(np.stack(next_states)).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(device)

        model.update(states, actions, next_states, rewards, dones)
        for states, actions, next_states, rewards, dones in data_loader:
            states = states.to(device)
            actions = actions.to(device)
            next_states = next_states.to(device)
            rewards = rewards.unsqueeze(1).to(device)
            dones = dones.unsqueeze(1).to(device)
            model.update(states, actions, next_states, rewards, dones)

        model.save('models/sac')
