#!/usr/bin/env python3
import random

import gym
import numpy as np
import torch
from tqdm import trange

from args import parse_args
import environments
from trainer import Trainer


def set_seed(seed: int):
    """Set all seeds

    Args:
        seed (int): Value of the seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)

    agent_cls = args.agent_cls
    trainer = Trainer(**args.train_args)

    all_returns = []
    for exp in trange(args.num):
        env = gym.make('{}-v0'.format(args.env), **args.env_args)
        if 'oracle' in args:
            args.agent_args['env'] = env
        elif 'embed' in args:
            args.agent_args['eps'] = trainer.eps
        agent = agent_cls(s_n=env.observation_space.n, a_n=env.action_space.n, **args.agent_args)
        returns = trainer.fit(agent, env)
        all_returns.append(returns)
    np.save('returns', np.array(all_returns))
