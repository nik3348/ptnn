import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.tensor(state).float().to(device),
            torch.tensor(action).float().to(device),
            torch.tensor(reward).float().unsqueeze(1).to(device),
            torch.tensor(next_state).float().to(device),
            torch.tensor(done).float().unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.max_action * torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, u):
        x = F.relu(self.fc1(torch.cat([x, u], 1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DDPG:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.001,
        actor_lr=1e-4,
        critic_lr=1e-4,
        batch_size=64,
        capacity=1000000,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.batch_size = batch_size

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(capacity)

    def select_action(self, state):
        state = torch.tensor(state).float().unsqueeze(0).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0 # Not enough transitions

        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size
        )

        with torch.no_grad():
            target_action = self.actor_target(next_state)
            target_value = self.critic_target(next_state, target_action)
            target = reward + (1 - done) * self.discount * target_value

        current_value = self.critic(state, action)
        critic_loss = F.mse_loss(current_value, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return critic_loss, actor_loss

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "/ddpg_actor.pth")
        torch.save(self.critic.state_dict(), filename + "/ddpg_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "/ddpg_actor.pth"))
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic.load_state_dict(torch.load(filename + "/ddpg_critic.pth"))
        self.critic_target.load_state_dict(self.critic.state_dict())


def start():
    env = gymnasium.make("MountainCarContinuous-v0", render_mode="rgb_array")
    writer = SummaryWriter()

    state_dim = env.observation_space.shape[0]
    action_dim = 1
    max_action = 1

    model = DDPG(state_dim, action_dim, max_action, batch_size=32)
    # model.load('models/ddpg')

    episodes = 500
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = model.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            model.store_transition(state, action, reward, next_state, done)
            critic_loss, actor_loss = model.train()
            state = next_state

        model.save('models/ddpg')
        writer.add_scalar("loss/critic", critic_loss, episode + 1)
        writer.add_scalar("loss/actor", actor_loss, episode + 1)
        writer.add_scalar("score/reward", episode_reward, episode + 1)
        print(f"Episode: {episode+1}/{episodes}, Reward: {episode_reward}")
