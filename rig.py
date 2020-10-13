# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import collections
import random

import gym
import numpy as np
import torch
from torch import optim
from torch.nn import functional as F

from models.bvae import BetaVAE
from models.ddpg import OUNoise
from models.sac import GaussianPolicy, QNetwork
from utils.memory import ReplayBuffer
from utils.sync import soft_sync

device = torch.device('cuda')


# %%
env = gym.make('FetchReach-v1')

# ## Vae training

# %%
pre_training = list()
pre_training.append(env.reset()['observation'])
for _ in range(1000):
    status = env.reset()
    state = status['observation']
    done = False
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        next_state = next_state['observation']
        pre_training.append(next_state)
        state = next_state


# %%
vae_model = BetaVAE(env.observation_space['observation'].shape[0],
                    env.observation_space['observation'].shape[0]//2)
vae_model = vae_model.to(torch.device('cuda'))
vae_optim = optim.Adam(vae_model.parameters(), lr=1e-3)
vae_model.train()


# %%
def train_vae(train_data):
    train_loss = 0
    for batch_idx, data in enumerate(train_data):
        data = torch.FloatTensor(data).to(torch.device('cuda'))
        vae_optim.zero_grad()
        results = vae_model(data)
        loss = vae_model.loss_function(*results, M_N=1/len(train_data))
        loss['loss'].backward()
        train_loss += loss['loss'].item()
        vae_optim.step()
    print('Train Loss:', train_loss/len(train_data))


for epoch in range(50):
    batches = np.array_split(np.array(pre_training), len(pre_training)//128)
    train_vae(batches)

# ## Training policy

# %%
alpha = 0.2
lr = 1e-3

target_entropy = - \
    torch.prod(torch.Tensor(env.action_space.shape).to(device)).item()
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optim = optim.Adam([log_alpha], lr=lr)

policy = GaussianPolicy((env.observation_space['observation'].shape[0]//2)*2,
                        env.action_space.shape[0]).to(device)

crt = QNetwork((env.observation_space['observation'].shape[0]//2)
               * 2, env.action_space.shape[0]).to(device)
tgt_crt = QNetwork((env.observation_space['observation'].shape[0]//2)*2,
                   env.action_space.shape[0]).to(device)

tgt_crt.load_state_dict(crt.state_dict())

policy_optim = optim.Adam(policy.parameters(), lr=lr)
crt_optim = optim.Adam(crt.parameters(), lr=lr)


# %%
noise = OUNoise(env.action_space)
memory = ReplayBuffer(1000000)


# %%
def dist(x, y):
    x = x.cpu().numpy()
    y = y.cpu().numpy()
    res = np.linalg.norm(x-y, axis=1)
    return torch.tensor(res).unsqueeze(1).to(device)


# %%
def train_policy(act_net, crt_net, tgt_crt_net,
                 optimizer_act, optimizer_crt,
                 memory, vae_model, batch_size=128,
                 automatic_entropy_tuning=True):
    global alpha, log_alpha, alpha_optim
    gamma = 0.99
    state_batch, action_batch, reward_batch,\
        next_state_batch, mask_batch, goal_batch = memory.sample(batch_size)

    state_batch = torch.FloatTensor(state_batch).to(device)
    goal_batch = torch.FloatTensor(goal_batch).to(device)
    next_state_batch = torch.FloatTensor(
        next_state_batch).to(device)
    action_batch = torch.FloatTensor(action_batch).to(device)

    reward_batch = torch.FloatTensor(
        reward_batch).to(device).unsqueeze(1)

    mask_batch = torch.BoolTensor(
        mask_batch).to(device).unsqueeze(1)

    state_batch, logvar = vae_model.encode(state_batch)
    state_batch, logvar = state_batch.detach(), logvar.detach()
    next_state_batch = vae_model.encode(next_state_batch)[0].detach()

    if np.random.rand() > 0.5:
        goal_batch = vae_model.reparameterize(state_batch, logvar)

    reward_batch = - dist(next_state_batch, goal_batch)

    state_batch = torch.cat((state_batch, goal_batch), 1)
    next_state_batch = torch.cat((next_state_batch, goal_batch), 1)

    with torch.no_grad():
        next_state_action, next_state_log_pi, _ = act_net.sample(
            next_state_batch)
        qf1_next_target, qf2_next_target = tgt_crt_net(
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
    qf1_loss = F.mse_loss(qf1, next_q_value.detach())
    # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    qf2_loss = F.mse_loss(qf2, next_q_value.detach())

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

    return policy_loss.item(), qf1_loss.item(), \
        qf2_loss.item(), alpha_loss.item()


def select_action(policy, state, evaluate=False):
    state = state.unsqueeze(0)
    if evaluate is False:
        action, _, _ = policy.sample(state)
    else:
        _, _, action = policy.sample(state)
    return action.detach().cpu().numpy()[0]


# %%
data_vae = collections.deque(maxlen=500000)
for data in pre_training:
    data_vae.append(data)

update_target = 0
for epi in range(1000):
    steps = 0
    state = env.reset()['observation']
    mu, logvar = vae_model.encode(torch.FloatTensor(state).to(device))
    mu, logvar = mu.detach(), logvar.detach()
    zg = vae_model.reparameterize(mu, logvar).detach()
    done = False
    episode = list()
    epi_reward = 0
    while not done:
        to_fwd = torch.cat((mu, zg))
        action = select_action(policy, to_fwd)
        next_state, reward, done, _ = env.step(action)
        next_state = next_state['observation']
        # if epi % 20 == 0:
        #     env.render()
        next_mu, _ = vae_model.encode(torch.FloatTensor(next_state).to(device))
        next_mu = next_mu.detach()
        memory.push(state, action, reward,
                    next_state, done, zg.cpu().numpy())
        episode.append((state, action, next_state, done))

        state = next_state
        data_vae.append(state)
        mu = next_mu
        update_target += 1
        steps += 1
        epi_reward += reward
    print('Episode', epi, '-> Reward:', epi_reward)
    if len(memory) > 128:
        for epoch in range(10):
            train_policy(policy, crt, tgt_crt, policy_optim,
                         crt_optim, memory, vae_model)
            soft_sync(crt, tgt_crt)

    for i, (state, action, next_state, done) in enumerate(episode):
        for t in np.random.choice(len(episode), 5):
            s_hi = episode[t][-2]
            s_hi, _ = vae_model.encode(
                torch.FloatTensor(next_state).to(device))
            s_hi = s_hi.detach().cpu().numpy()
            memory.push(state, action, 0, next_state, done, s_hi)

    if epi % 10 == 0 and epi > 0:
        batches = [random.sample(data_vae, 128) for _ in range(10)]
        for _ in range(10):
            train_vae(batches)
