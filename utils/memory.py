import collections
import random

import torch


class ReplayBuffer():
    def __init__(self, buffer_limit=1000000):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        state_batch, action_batch, reward_batch,\
            next_state_batch, done_mask_batch = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            state_batch.append(s)
            action_batch.append([a])
            reward_batch.append([r])
            next_state_batch.append(s_prime)
            done_mask_batch.append([done_mask])

        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.tensor(action_batch)
        reward_batch = torch.tensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_mask_batch = torch.tensor(done_mask_batch)

        return state_batch, action_batch, reward_batch,\
            next_state_batch, done_mask_batch

    def size(self):
        return len(self.buffer)
