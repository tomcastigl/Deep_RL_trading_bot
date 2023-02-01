import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import random
import yaml
import mplfinance as mpl
import seaborn as sns
from tqdm import tqdm, trange
import argparse
import warnings

from models import (
    ReplayMemory,
    agent,
    simple_feed_forward,
    simple_convnet,
    dec_exp,
    compute_perf,
    compute_rews,
)


def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(
                f"Command line argument '{key}' (value: "
                f"{arg_dict[key]}) will be overwritten with value "
                f"{value} provided in the config file."
            )
        if isinstance(value, dict):
            arg_dict[key] = argparse.Namespace(**value)
        else:
            arg_dict[key] = value

    return args


def merge_configs(config, resume_config):
    for key, value in resume_config.items():
        if isinstance(value, argparse.Namespace):
            value = value.__dict__
        if key in config and config[key] != value:
            warnings.warn(
                f"Config parameter '{key}' (value: "
                f"{config[key]}) will be overwritten with value "
                f"{value} from the checkpoint."
            )
        config[key] = value
    return config


# ------------------------------------------------------------------------------
# Training
# ______________________________________________________________________________
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--config", type=str, required=True)
    p.add_argument("--resume", type=str, default=None)
    args = p.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Get main config
    ckpt_path = None if args.resume is None else Path(args.resume)
    if args.resume is not None:
        resume_config = torch.load(ckpt_path, map_location=torch.device("cpu"))[
            "hyper_parameters"
        ]

        config = merge_configs(config, resume_config)

    args = merge_args_and_yaml(args, config)

    Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))

    # data to be found at https://drive.google.com/drive/u/0/folders/1Eio-9xGLze3RlNCrJn1B3bzAGLxwyDis

    train_data_path = "data/train_" + args.timeframe + args.currency + ".npy"
    test_data_path = "data/test_" + args.timeframe + args.currency + ".npy"
    if args.train_N_sample != None:
        data = torch.from_numpy(np.load(train_data_path))[: args.train_N_sample]
    else:
        data = torch.from_numpy(np.load(train_data_path))
    data_test = torch.from_numpy(np.load(test_data_path))

    print(f"train data shape: {data.shape}")
    print(f"{args.N_candles=}")

    # training
    mask = np.zeros(data.shape[0], dtype=bool)
    memory = ReplayMemory(args.mem_len)
    t = 0
    if args.total_steps != None:
        steps = args.total_steps
    else:
        steps = data.shape[0] - args.N_candles
    in_pos = False
    agent = agent()

    if args.model == "feed-forward":
        policy = simple_feed_forward(args.N_candles)  # model
        target = simple_feed_forward(args.N_candles)  # target
    elif args.model == "convnet":
        policy = simple_convnet()  # model
        target = simple_convnet()  # target
    else:
        raise Exception("bad model specification")

    target.load_state_dict(policy.state_dict())
    target.eval()
    criterion = torch.nn.HuberLoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    gamma = args.gamma
    batch_size = args.batch_size
    losses = []
    rewards_ = []
    results_ = []
    times = []
    sa_means = []
    actions_ = []
    count = 0

    for i in tqdm(range(args.N_candles, steps), unit="step", desc="training"):
        agent.epsilon = dec_exp(i, steps // 2)
        # if i %(steps//10)==0:
        #    print(agent.epsilon)
        #    print(f'step {i} / {steps}')
        #    print('n of pos last block', count)
        #    print(f'do nothing {round(actions_[-(steps//10):].count(0)/(steps//10),3)},'+
        #            f'long {round(actions_[-(steps//10):].count(1)/(steps//10),3)},'+
        #            f'short {round(actions_[-(steps//10):].count(2)/(steps//10),3)}')
        #    count=0
        state = torch.zeros(3, args.N_candles, data.shape[1])
        mask = np.zeros_like(mask)
        mask[i - args.N_candles : i] = 1
        state[agent.position] = data[mask]

        # take action
        action = agent.act(state.unsqueeze(0), policy)
        actions_.append(action)
        if (in_pos == False) and (action != 0):
            time = i
            in_pos = True
            agent.position = action
            agent.entry_price = state[:, -1, 3].sum()

        if (in_pos == True) and (action == agent.position):
            agent.last_price = state[:, -1, 3].sum()

        if (in_pos == True) and (action not in [0, agent.position]):
            in_pos = False
            times.append(i - time)
            short = bool(agent.position - 1)
            reward = compute_rews(agent.entry_price, state[:, -1, 3].sum(), short)
            result = compute_perf(agent.entry_price, state[:, -1, 3].sum(), short)
            results_.append(result.detach().numpy())
            rewards_.append(reward)
            agent.position = 0

            next_state = torch.zeros(3, args.N_candles, data.shape[1])
            next_mask = np.zeros_like(mask)
            next_mask[t + 1 : t + args.N_candles + 1] = 1
            next_state[agent.position] = data[next_mask]
            memory.push(
                state, torch.tensor([action]), next_state, torch.tensor([reward])
            )
            count += 1

        else:
            next_state = torch.zeros(3, args.N_candles, data.shape[1])
            next_mask = np.zeros_like(mask)
            next_mask[t + 1 : t + args.N_candles + 1] = 1
            next_state[agent.position] = data[next_mask]
            memory.push(state, torch.tensor([action]), next_state, torch.tensor([0]))

        # -----------udpt------------#
        if len(memory) > batch_size:
            sample = memory.sample(batch_size)
            sample = Transition(*zip(*sample))

            non_final_mask = torch.tensor(
                tuple(map(lambda s: s is not None, sample.next_state))
            )
            non_final_new_states = torch.cat(
                [s.unsqueeze(0) for s in sample.next_state if s is not None]
            )
            states = torch.cat([state.unsqueeze(0) for state in sample.state])
            actions = torch.cat(sample.action)
            rewards = torch.cat(sample.reward)
            s_a_vals = policy(states).gather(1, actions.unsqueeze(0))

            new_state_vals = torch.zeros(batch_size)

            new_state_vals[non_final_mask] = target(non_final_new_states).max(1)[0]
            sa_means.append(s_a_vals.mean().detach().numpy())

            expected = new_state_vals * gamma + rewards
            loss = criterion(s_a_vals, expected.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            for param in policy.parameters():
                param.grad.data.clamp_(-1, 1)
            losses.append(float(loss.detach()))
            optimizer.step()

        if i % args.target_update_step == 0:  # update target
            target.load_state_dict(policy.state_dict())

        t += 1
    print("saving model ...")
    torch.save(policy.state_dict, "model.pt")
    plt.title("training loss")
    plt.plot(range(len(losses)), losses)
    plt.savefig("train_loss.png", dpi=300)

    print("\nactions taken ratio:")
    print(
        f"do nothing {round(actions_.count(0)/steps,3)}\t"
        + f"long {round(actions_.count(1)/steps,3)}\t"
        + f"short {round(actions_.count(2)/steps,3)}\t"
    )
    print(20 * "-", "done", 20 * "-")
