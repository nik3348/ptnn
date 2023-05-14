from itertools import count
import gymnasium as gym
import matplotlib
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

def ppo2(env_name, num_episodes):
    env = gym.make(env_name, render_mode="human")
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    model = ActorCritic(state_size, action_size).to("cuda")
    # model.load_state_dict(torch.load('model.pth'))

    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    eps_clip = 0.2
    gamma = 0.99
    t_horizon = 2048
    results = []

    def plot_durations(show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(results, dtype=torch.float)
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
    
    for ep_no in range(num_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        log_probs = []
        values = []
        rewards = []
        
        for t in range(t_horizon):
            state = torch.FloatTensor(state).to("cuda")
            actor_logits, critic_value = model(state)
            dist = Categorical(logits=actor_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = critic_value.item()
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            episode_reward += reward
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            if done:
                results.append(t)
                break
                
            state = next_state
            
        next_value = 0
        if not done:
            state = torch.FloatTensor(state)
            _, next_value = model(state)
            next_value = next_value.item()
        
        returns = []
        advantages = []
        return_so_far = next_value
        
        for t in reversed(range(len(rewards))):
            return_so_far = rewards[t] + gamma * return_so_far
            advantage = return_so_far - values[t]
            returns.insert(0, return_so_far)
            advantages.insert(0, advantage)
        
        advantages = torch.FloatTensor(advantages).to("cuda")
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        returns = torch.FloatTensor(returns)
        log_probs = torch.stack(log_probs)
        
        values = torch.FloatTensor(values)
        clipped_values = values + torch.clamp(returns - values, -eps_clip, eps_clip)
        value_loss = nn.functional.mse_loss(returns, values)
        clipped_value_loss = nn.functional.mse_loss(returns, clipped_values)
        
        policy_ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = policy_ratio * advantages
        surr2 = torch.clamp(policy_ratio, 1-eps_clip, 1+eps_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        loss = policy_loss + 0.5 * value_loss + 0.5 * clipped_value_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        plot_durations()
        print(f"Episode: {ep_no+1} Score: {episode_reward}")

    env.close()
    plot_durations(True)
    torch.save(model.state_dict(), 'model.pth')


def start():
    env_name = "CartPole-v1"
    num_episodes = 1000
    ppo2(env_name, num_episodes)

