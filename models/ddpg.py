import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Critic(nn.Module):
    def __init__(self, num_inputs):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Actor(nn.Module):
    def __init__(self, num_inputs, action_size):
        super(Actor, self).__init__()
        self.num_input = num_inputs
        self.linear1 = nn.Linear(num_inputs, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, action_size)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.from_numpy(state).float().to(device)
        state = state.view(1, self.num_input)
        action = self.forward(state)
        return action.detach().cpu().numpy()


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15,
                 max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - \
            (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


def train(critic, critic_target, actor, actor_target,
          critic_optim, actor_optim, memory, batch_size):

    state_batch, action_batch,\
        reward_batch, next_state_batch, done_batch = memory.sample(batch_size)

    state_batch = state_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    action_batch = action_batch.to(device).squeeze()
    reward_batch = reward_batch.to(device)
    done_batch = done_batch.to(device)

    actor_loss = critic(state_batch, actor(state_batch))
    actor_loss = -actor_loss.mean()

    next_actions_target = actor_target(next_state_batch)
    q_targets = critic_target(next_state_batch, next_actions_target)
    targets = reward_batch + (1.0 - done_batch)*gamma*q_targets

    q_values = critic(state_batch, action_batch)
    critic_loss = F.smooth_l1_loss(q_values, targets.detach())
    critic_optim.zero_grad()
    critic_loss.backward()
    critic_optim.step()

    actor_optim.zero_grad()
    actor_loss.backward()
    actor_optim.step()

    return actor_loss.item(), critic_loss.item()
