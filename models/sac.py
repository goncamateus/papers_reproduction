import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256):
        super(QNetwork, self).__init__()

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x2 = F.relu(self.linear2(x1))
        x3 = self.linear3(x2)

        x4 = F.relu(self.linear4(xu))
        x5 = F.relu(self.linear5(x4))
        x6 = self.linear6(x5)

        return x3, x6


class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256, action_space=None):
        super(GaussianPolicy, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x1 = F.relu(self.linear1(state))
        x2 = F.relu(self.linear2(x1))
        mean = self.mean_linear(x2)
        log_std = self.log_std_linear(x2)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        with torch.autograd.set_detect_anomaly(True):
            mean, log_std = self.forward(state)
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob = log_prob -\
                torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(GaussianPolicy, self).to(device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)


def train(act_net, crt_net, tgt_crt_net,
          optimizer_act, optimizer_crt,
          memory, batch_size, automatic_entropy_tuning=True):
    state_batch, action_batch, reward_batch,\
        next_state_batch, mask_batch = memory.sample(batch_size)

    state_batch = state_batch.to(device)
    next_state_batch = next_state_batch.to(device)
    action_batch = action_batch.to(device).squeeze()
    reward_batch = reward_batch.to(device)
    mask_batch = done_batch.to(device)

    with torch.no_grad():
        next_state_action, next_state_log_pi, _ = act_net.sample(
            next_state_batch)
        qf1_next_target, qf2_next_target = tgt_crt_net.target_model(
            next_state_batch, next_state_action)
        min_qf_next_target = torch.min(
            qf1_next_target,
            qf2_next_target) - alpha * next_state_log_pi
        min_qf_next_target[mask_batch] = 0.0
        next_q_value = reward_batch + gamma * (min_qf_next_target)
    # Two Q-functions to mitigate
    # positive bias in the policy improvement step
    qf1, qf2 = crt_net(state_batch, action_batch)
    # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    qf1_loss = F.mse_loss(qf1, next_q_value)
    # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    qf2_loss = F.mse_loss(qf2, next_q_value)

    pi, log_pi, _ = act_net.sample(state_batch)

    qf1_pi, qf2_pi = crt_net(state_batch, pi)
    min_qf_pi = torch.min(qf1_pi, qf2_pi)

    # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
    policy_loss = ((alpha * log_pi) - min_qf_pi).mean()

    optimizer_crt.zero_grad()
    qf1_loss.backward()
    optimizer_crt.step()

    optimizer_crt.zero_grad()
    qf2_loss.backward()
    optimizer_crt.step()

    optimizer_act.zero_grad()
    policy_loss.backward()
    optimizer_act.step()

    if automatic_entropy_tuning:
        alpha_loss = -(log_alpha * (log_pi +
                                    target_entropy
                                    ).detach()).mean()

        alpha_optim.zero_grad()
        alpha_loss.backward()
        alpha_optim.step()

        alpha = log_alpha.exp()
    else:
        alpha_loss = torch.tensor(0.).to(device)

    return policy_loss.item(), qf1_loss.item(),\
        qf2_loss.item(), alpha_loss.item()
