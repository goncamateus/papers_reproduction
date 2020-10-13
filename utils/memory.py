import collections
import random

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, seed=123456):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done, goal):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward,
                                      next_state, done, goal)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, goal = map(
            np.stack, zip(*batch))
        return state, action, reward, next_state, done, goal

    def __len__(self):
        return len(self.buffer)
