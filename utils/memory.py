import collections
import random

import torch


class ReplayBuffer():
    def __init__(self, buffer_limit=1000000):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        device = torch.device('cuda')
        mini_batch = random.sample(self.buffer, n)
        state_batch, action_batch, reward_batch = [], [], []
        next_state_batch, done_mask_batch, goal_batch = [], [], []
        with_goal = False

        for transition in mini_batch:
            if len(transition) < 6:
                s, a, r, s_prime, done_mask = transition
                state_batch.append(s)
                action_batch.append([a])
                reward_batch.append([r])
                next_state_batch.append(s_prime)
                done_mask_batch.append([done_mask])
            else:
                with_goal = True
                s, a, r, s_prime, done_mask, goal = transition
                state_batch.append(s)
                action_batch.append([a])
                reward_batch.append([r])
                next_state_batch.append(s_prime)
                done_mask_batch.append([done_mask])
                goal_batch.append(goal)

        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.tensor(reward_batch).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_mask_batch = torch.tensor(done_mask_batch).to(device)
        res = [state_batch, action_batch, reward_batch,
               next_state_batch, done_mask_batch]
        if with_goal:
            goal_batch = torch.FloatTensor(goal_batch).to(device)
            res.append(goal_batch)

        return res

    def size(self):
        return len(self.buffer)
