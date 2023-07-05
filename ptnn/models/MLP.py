import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


class Actor(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_mean = nn.Linear(hidden_dim, action_dim)
        self.action_std = nn.Linear(hidden_dim, action_dim)
        self.initialize_weights()

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        mean = self.action_mean(x)
        std = torch.exp(self.action_std(x))
        return mean, std

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight.data)
                constant_(m.bias.data, 0)


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.initialize_weights()

    def forward(self, x, u):
        x = F.leaky_relu(self.fc1(torch.cat([x, u], 1)))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_uniform_(m.weight.data)
                constant_(m.bias.data, 0)


class ActorCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim, action_dim):
        super(ActorCritic, self).__init__()

        self.actor = Actor(input_dim, hidden_dim, action_dim)
        self.critic = Critic(input_dim, hidden_dim, action_dim)

    def forward(self, state):
        state = state.view(state.size(0), -1)

        action_mean, action_std = self.actor(state)
        dist = torch.distributions.Normal(action_mean, action_std)

        value = self.critic(state, action_mean)

        return dist, value
 
    def save(self, filename):
        torch.save(self.actor.state_dict(), filename + "/ppo_actor.pth")
        torch.save(self.critic.state_dict(), filename + "/ppo_critic.pth")

    def load(self, filename):
        self.actor.load_state_dict(torch.load(filename + "/ppo_actor.pth"))
        self.critic.load_state_dict(torch.load(filename + "/ppo_critic.pth"))
