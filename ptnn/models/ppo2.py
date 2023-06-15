import random
import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from ptnn.wrappers.ppo2Wrapper import Wrapper
from ptnn.layers.PositionalEncoding import PositionalEncoding
from ptnn.layers.Encoder import Encoder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        super(Actor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.act1 = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)

        self.mlp_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        ])

        self.skip_connection = nn.Linear(input_size, hidden_size)
        self.act2 = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        h0 = nn.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
        c0 = nn.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
        nn.init.xavier_normal_(h0)
        nn.init.xavier_normal_(c0)

        outputs, _ = self.lstm(x, (h0, c0))
        outputs = self.act1(outputs)
        outputs = self.layer_norm(outputs)
        outputs = self.dropout_layer(outputs)

        x = self.skip_connection(x)
        x = self.act2(x)
        outputs = outputs + x

        outputs = torch.mean(outputs, dim=1)
        for layer in self.mlp_layers:
            outputs = layer(outputs)

        return outputs


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super(Critic, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.act1 = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout_layer = nn.Dropout(dropout)

        self.mlp_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        ])

        self.skip_connection = nn.Linear(input_size, hidden_size)
        self.act2 = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        for layer in self.mlp_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        h0 = nn.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
        c0 = nn.Parameter(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device))
        nn.init.xavier_normal_(h0)
        nn.init.xavier_normal_(c0)

        outputs, _ = self.lstm(x, (h0, c0))
        outputs = self.act1(outputs)
        outputs = self.layer_norm(outputs)
        outputs = self.dropout_layer(outputs)

        x = self.skip_connection(x)
        x = self.act2(x)
        outputs = outputs + x

        outputs = torch.mean(outputs, dim=1)
        for layer in self.mlp_layers:
            outputs = layer(outputs)

        return outputs


class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size,  num_layers, action_size):
        super(ActorCritic, self).__init__()
        self.actor = Actor(input_size, hidden_size, num_layers, action_size)
        self.critic = Critic(input_size, hidden_size, num_layers)

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value


input_size = 4
hidden_size = 64
num_layers = 2
output = 2
def ppo2(env_name):
    # Set hyperparameters
    num_epochs = 1000
    num_steps = 1024
    mini_batch_epochs = 10
    mini_batch_size = 128
    learning_rate = 1e-4
    gamma = 0.4
    clip_param = 0.2
    value_coeff = 0.5
    entropy_coeff = 0.01
    max_seq_len = 5
    exploration_rate = 0.00001
    noise_std_dev = 0.0001
    max_grad_norm = 2.0

    # Create the environment
    env = gymnasium.make(env_name, max_episode_steps=num_steps, render_mode="human")
    env = Wrapper(env, max_seq_len)

    # Initialize the model and optimizer
    # env.observation_space.shape[0]
    # env.action_space.n
    model = ActorCritic(input_size, hidden_size, num_layers, output).to(device)
    model.load_state_dict(torch.load('models/model.pth'))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    writer = SummaryWriter()

    # Training loop
    max_score = 0
    episode_durations = []
    offline = list()
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
                actor_output, value = model(state)
                action_probs = torch.softmax(actor_output, dim=1)
            dist = Categorical(logits=action_probs)

            if random.uniform(0, 1) < exploration_rate:
                action = torch.tensor([env.action_space.sample()], dtype=torch.long).to(device)
            else:
                action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.item())
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
                writer.add_scalar("Score/Actual", score, epoch)
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
        if episode_durations[epoch] > torch.FloatTensor(episode_durations).mean():
            offline += dataset
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

        # Calculate gradient norms
        norms = []
        for param in model.parameters():
            if param.grad is not None:
                norm = torch.norm(param.grad.data)
                norms.append(norm.item())

        # Store the average gradient norm for this iteration
        average_norm = sum(norms) / len(norms)
        writer.add_scalar("Gradient/AvgNorm", average_norm, epoch)

        if epoch % 100 == 0:
            torch.save(offline, 'data/offline.pth')
            torch.save(model.state_dict(), 'models/model.pth')

        if epoch > 100:
            writer.add_scalar(
                "Score/Average", torch.FloatTensor(episode_durations).mean().item(), epoch)
        writer.add_scalar("Loss/Policy", policy_loss.item(), epoch)
        writer.add_scalar("Loss/Value", value_loss.item(), epoch)
        writer.add_scalar("Loss/Entropy", entropy_loss.item(), epoch)
        writer.add_scalar("Loss/Total", total_loss.item(), epoch)
    writer.close()


def train():
    epochs = 10
    learning_rate = 1e-5
    clip_param = 0.2
    value_coeff = 0.5
    entropy_coeff = 0.01
    noise_std_dev = 0.001

    model = ActorCritic(input_size, hidden_size, num_layers, output).to(device)
    # model.load_state_dict(torch.load('models/model.pth'))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Loading Data')
    data_loader = DataLoader(torch.load('data/offline-200.pth'), batch_size=128, shuffle=True)

    print('Training')
    for t in range(epochs):
        print(t + 1)

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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            torch.save(model.state_dict(), 'models/model.pth')
