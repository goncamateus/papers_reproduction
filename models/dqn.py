import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.01


class Qnet(nn.Module):
    def __init__(self, num_input, actions):
        super(Qnet, self).__init__()
        self.actions = actions
        self.num_input = num_input
        self.fc1 = nn.Linear(num_input, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def sample_action(self, obs, epsilon):
        obs = torch.from_numpy(obs).float().to(device)
        obs = obs.view(1, self.num_input)
        out = self.forward(obs)
        coin = random.random()
        if coin < epsilon:
            return random.randint(0, self.actions-1)
        else:
            return out.argmax().item()


def train(q, q_target, memory, optimizer, batch_size):
    losses = list()
    state_batch, action_batch,\
        reward, next_state_batch, done_mask = memory.sample(batch_size)
    state_batch = state_batch.to(device)
    action_batch = action_batch.to(device)
    reward = reward.to(device)
    next_state_batch = next_state_batch.to(device)
    done_mask = done_mask.to(device)

    q_out = q(state_batch)
    q_a = q_out.gather(1, action_batch)
    max_q_prime = q_target(next_state_batch).max(1)[0].unsqueeze(1)
    target = reward + GAMMA * max_q_prime * done_mask
    loss = F.smooth_l1_loss(q_a, target)

    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return losses
