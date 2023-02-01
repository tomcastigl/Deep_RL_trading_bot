import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
from tqdm import tqdm
import random

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# helper function for decreasing exploration rate.
def dec_exp(n, n_, e_min=0.1, e_max=0.8):
    return max(e_min, e_max * (1 - n / n_))


# agent class
class agent:
    def __init__(self, money=0, epsilon=0):
        # 0=do-nothing, 1=long, 2=short
        self.position = 0
        self.entry_price = None
        self.money = money
        self.epsilon = epsilon

    # epsilon-greedy policy
    def act(self, state, policy):
        if np.random.random() > self.epsilon:
            action = int(torch.argmax(policy(state)))
        else:
            action = np.random.choice([0, 1, 2])
        return action


def compute_rews(entry_price, last_price, short: bool):
    key = {True: -1, False: 1}
    rew = key[short] * int(last_price < entry_price)
    return rew


def compute_perf(entry_price, last_price, short: bool):
    key = {True: -1, False: 1}
    res = key[short] * ((last_price - entry_price) / entry_price)
    return res


class simple_feed_forward(
    nn.Module
):  # test a CNN then other cool stuff, then try smth like muzero
    def __init__(self, N_CANDLES):
        super(simple_feed_forward, self).__init__()
        self.n_candles = N_CANDLES
        self.fc = nn.Sequential(
            nn.Linear(3 * self.n_candles * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = x.view(-1, 3 * self.n_candles * 5)
        x = self.fc(x)
        return x


class simple_convnet(nn.Module):
    def __init__(self):
        super(simple_convnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, (5, 5))
        self.conv2 = nn.Conv2d(32, 32, (5, 1))
        self.fc1 = nn.Linear(2944, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)

        return x


# test cumulative return function
def test_cumul(policy, data):
    policy.eval()
    agent = agent(money=1000, epsilon=0)
    mask = np.zeros(data.shape[0], dtype=bool)
    in_pos = False
    pf = [agent.money]
    trades = []
    actions = []
    print(data.shape[0])
    for i in tqdm(range(policy.n_candles, data.shape[0]), unit="step", desc="testing"):
        state = torch.zeros(3, policy.n_candles, data.shape[1])
        mask = np.zeros_like(mask)
        mask[i - policy.n_candles : i] = 1
        state[agent.position] = data[mask]
        action = int(torch.argmax(policy(state.unsqueeze(0))))
        actions.append(action)
        if (in_pos == False) and (action != 0):
            in_pos = True
            agent.position = action
            agent.entry_price = state[:, -1, 3].sum()

        if (in_pos == True) and (action == agent.position):
            agent.last_price = state[:, -1, 3].sum()

        if (in_pos == True) and (action not in [0, agent.position]):
            in_pos = False
            short = bool(agent.position - 1)
            result = compute_perf(
                agent.entry_price, state[:, -1, 3].sum(), short
            ).numpy()
            agent.money += agent.money * result

            trades.append(
                (
                    i,
                    float(agent.entry_price),
                    float(state[:, -1, 3].sum()),
                    short,
                    float(result),
                )
            )
            agent.position = 0
        pf.append(agent.money)
    return (
        pf,
        pd.DataFrame(trades, columns=["idx_data", "entry", "close", "short", "res"]),
        actions,
    )
