import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as dist
from ptnn.models.LSTM import ActorCritic
from pettingzoo.classic import tictactoe_v3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def start():
    # Hyperparameters
    learning_rate = 0.001
    discount_factor = 0.99
    clip_range = 0.2
    num_mini_batches = 4

    env = tictactoe_v3.env(render_mode="rgb_array")

    # Define the training parameters
    num_epochs = 100
    learning_rate = 0.001

    input_size = 9
    hidden_size = 16
    num_layers = 9
    output_size = env.action_spaces["player_1"].n
    model = ActorCritic(input_size, hidden_size, num_layers, output_size).to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0
    memory_agent1 = []
    memory_agent2 = []

    observations = []
    actions = []
    rewards = []
    values = []
    sequence_lengths = []

    for epoch in range(num_epochs):
        env.reset()
        for agent in env.agent_iter():
            steps += 1
            observation, reward, termination, truncation, info = env.last()

            if termination or truncation:
                break

            obs = observation["observation"]
            mask = observation["action_mask"]

            cur = []
            for x in obs:
                for y in x:
                    cur.append(y[0] - y[1])

            if agent == "player_1":
                memory_agent1.append(cur)
                memory = memory_agent1
            else:
                memory_agent2.append(cur)
                memory = memory_agent2

            obs = torch.FloatTensor(memory).unsqueeze(0).to(device)

            with torch.no_grad():
                action_logits, value = model(obs, [len(memory)])
            action_logits = action_logits + (1 - torch.FloatTensor(mask).to(device)) * -1e9
            action_dist = dist.Categorical(logits=action_logits)
            action = action_dist.sample()
            env.step(action.item())

            observations.append(obs.squeeze(0))
            actions.append(action.cpu().detach().numpy())
            rewards.append(reward)
            values.append(value.item())
            sequence_lengths.append(len(memory))

        # Compute rewards and advantages
        rewards_to_go = []
        advantage_estimates = []
        discounted_reward = 0
        next_value = 0
        for t in reversed(range(len(rewards))):
            discounted_reward = rewards[t] + discount_factor * next_value
            rewards_to_go.insert(0, discounted_reward)
            advantage_estimates.insert(0, discounted_reward - next_value)
            next_value = rewards_to_go[0]

        # Normalize advantages
        advantages_tensor = torch.FloatTensor(advantage_estimates).to(device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # padded_observations = sorted(padded_observations, key=lambda x: len(x), reverse=True)
        padded_observations = nn.utils.rnn.pad_sequence(observations, batch_first=True, padding_value=0)

        actions_tensor = torch.LongTensor(numpy.array(actions)).to(device)
        values_tensor = torch.FloatTensor(values).to(device)

        action_logits, state_values = model(padded_observations, sequence_lengths)
        action_logits = action_logits.squeeze()

        # Compute policy loss
        action_dist = dist.Categorical(logits=action_logits)
        log_probs = action_dist.log_prob(actions_tensor)
        ratio = torch.exp(log_probs - log_probs.detach())
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = nn.MSELoss()(state_values.squeeze(), torch.FloatTensor(rewards_to_go).to(device))
        entropy_loss = -torch.mean(action_dist.entropy())
        total_loss = policy_loss + value_loss - entropy_loss

        # Perform optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Clear the memory buffers
        observations = []
        actions = []
        rewards = []
        values = []
        sequence_lengths = []

        # Print training progress
        if steps % 100 == 0:
            print(f"Steps: {steps}, Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}, "
                f"Entropy Loss: {entropy_loss.item()}")

        print('Reward: ', reward)
        torch.cuda.empty_cache()
    env.close()
