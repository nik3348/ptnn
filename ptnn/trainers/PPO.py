import random
import gymnasium
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ptnn.models.MLP import ActorCritic
from torch.utils.tensorboard import SummaryWriter

from ptnn.utils.logger import reward_metric


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 2
hidden_size = 256
output = 1
num_layers = 2

folder = 'checkpoints/ppo'
model = ActorCritic(input_size, hidden_size, output).to(device)
# model.load(folder)

def PPO(env_name):
    # Set hyperparameters
    num_epochs = 200
    num_steps = 2000
    mini_batch_size = 512
    gamma = 0.99
    max_seq_len = 1
    exploration_rate = 1e-4

    # Create the environment
    env = gymnasium.make(env_name, render_mode="rgb_array", max_episode_steps=num_steps)
    writer = SummaryWriter()

    # Training loop
    episode_durations = []
    for epoch in range(num_epochs):
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []

        next_states = []
        dones = []

        state, _ = env.reset()
        for _ in range(num_steps):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                dist, value = model(state)

            if random.uniform(0, 1) < exploration_rate:
                action = torch.tensor([env.action_space.sample()]).to(device)
            else:
                action = dist.sample()

            next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
            done = terminated or truncated
            log_prob = dist.log_prob(action)

            states.append(state.squeeze(dim=0))
            actions.append(action.squeeze(dim=0))
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob.squeeze(dim=0))

            next_states.append(next_state)
            dones.append(done)

            state = next_state

            if done:
                total_reward = torch.FloatTensor(rewards).sum()
                episode_durations.append(total_reward)
                writer.add_scalar("Score/Actual", total_reward, epoch)
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
        advantages = advantages.unsqueeze(1)

        # Update policy
        dataset = list(zip(states, actions, log_probs, returns, advantages))
        data_loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=True)
        policy_loss, value_loss, entropy_loss, total_loss = train(data_loader)

        # Calculate gradient norms
        norms = []
        for param in model.parameters():
            if param.grad is not None:
                norm = torch.norm(param.grad.data)
                norms.append(norm.item())

        # Store the average gradient norm for this iteration
        average_norm = sum(norms) / len(norms)
        writer.add_scalar("Gradient/AvgNorm", average_norm, epoch)

        if epoch % 50 == 0 and epoch > 0:
            print('Checkpointing')
            model.save(folder)

        if epoch > 100:
            writer.add_scalar("Score/Average", torch.FloatTensor(episode_durations).mean().item(), epoch)
        writer.add_scalar("Loss/Policy", policy_loss.item(), epoch)
        writer.add_scalar("Loss/Value", value_loss.item(), epoch)
        writer.add_scalar("Loss/Entropy", entropy_loss.item(), epoch)
        writer.add_scalar("Loss/Total", total_loss.item(), epoch)
        reward_metric(epoch, num_epochs, total_reward)
    writer.close()


def train(
        data_loader=None,
        epochs=10,
        learning_rate=1e-4,
        clip_param=0.2,
        value_coeff=0.5,
        entropy_coeff=0.01,
        noise_std_dev=0.0001,
        max_grad_norm=2.0,
    ):

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if data_loader is None:
        print('Loading Data')
        data_loader = DataLoader(torch.load('data/offline.pth'), batch_size=128, shuffle=True)
        print('Training')

    for _ in range(epochs):
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

            dist, value_pred = model(states)
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

            model.save(folder)

    return policy_loss, value_loss, entropy_loss, total_loss
