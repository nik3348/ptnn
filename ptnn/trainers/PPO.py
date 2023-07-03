import random
import gymnasium
import torch
import torch.nn.functional as F
import torch.optim as optim
from ptnn.wrappers.ppo2Wrapper import Wrapper
from ptnn.models.MLP import ActorCritic
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from supersuit import color_reduction_v0, frame_stack_v1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = 8
hidden_size = 512
output = 2
num_layers = 2

model = ActorCritic(input_size, hidden_size, output).to(device)
# model.load_state_dict(torch.load('models/model.pth'))

def ppo2(env_name):
    # Set hyperparameters
    num_epochs = 1000
    num_steps = 5000
    mini_batch_size = 512
    gamma = 0.2
    max_seq_len = 1
    exploration_rate = 1e-5

    # Create the environment
    env = gymnasium.make(env_name, render_mode="rgb_array", continuous=True)

    writer = SummaryWriter()

    # Training loop
    max_score = 0
    episode_durations = []
    offline = []
    # foo = torch.load('data/offline.pth')
    for epoch in range(num_epochs):
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []

        next_states = []
        dones = []

        save = False
        state, _ = env.reset()
        for score in range(num_steps):
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
                foo = torch.FloatTensor(rewards).sum()
                episode_durations.append(foo)
                writer.add_scalar("Score/Actual", foo, epoch)
                # if score > max_score:
                #     max_score = score
                #     writer.add_scalar("Score/Max", max_score, epoch)

                # if score < num_steps - 1:
                #     save = True
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
        if save:
            offline += dataset
            save = False
            print('Saving Data')

        if offline:
            bar = DataLoader(offline, batch_size=mini_batch_size, shuffle=True)
            policy_loss, value_loss, entropy_loss, total_loss = train(bar)

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

        if epoch % 100 == 0:
            print('Checkpointing')
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


def train(data_loader=None):
    epochs = 10
    learning_rate = 1e-4
    clip_param = 0.2
    value_coeff = 0.5
    entropy_coeff = 0.01
    noise_std_dev = 0.0001
    max_grad_norm = 2.0
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if data_loader is None:
        print('Loading Data')
        data_loader = DataLoader(torch.load('data/offline.pth'), batch_size=128, shuffle=True)
        print('Training')

    for t in range(epochs):
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

            torch.save(model.state_dict(), 'models/model.pth')

    return policy_loss, value_loss, entropy_loss, total_loss
