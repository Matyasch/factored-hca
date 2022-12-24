from argparse import ArgumentParser, RawTextHelpFormatter
from functools import partial
import sys

import agents


def agent_cls(agent, embed, factored, variant, oracle):
    if agent == 'PG':
        if embed:
            return agents.embedding.EmbeddingPG
        else:
            return agents.lookup.LookupPG
    else:  # SHCA
        if oracle:
            return agents.lookup.OracleSHCA
        elif embed:
            if factored:
                return agents.embedding.FactoredPairSHCA
            elif variant == 'shared':
                return agents.embedding.SharedSHCA
            elif variant == 'separate':
                return agents.embedding.SeparateSHCA
            elif variant == 'pair':
                return agents.embedding.PairSHCA
            else:
                raise ValueError('Please choose a variant of EmbeddinSHCA to use')
        else:
            if factored:
                return agents.lookup.FactoredLookupSHCA
            else:
                return agents.lookup.LookupSHCA


def parse_args():
    parser = ArgumentParser(description='Please note, that some combinations of arguments lead to undefined behaviour.')
    parser.formatter_class = partial(RawTextHelpFormatter, max_help_position=30)
    parser.add_argument('-n', '--num', type=int, default=1000,
                        help='Number of times to repeat training')
    parser.add_argument('-s', '--seed', type=int, default=0,
                        help='Seed to set before starting the experiments (once).')
    parser.add_argument('-a', '--agent', choices=['PG', 'SHCA'], required=True, metavar='AGENT',
                        help='Can be PG for Ploicy Gradient or SHCA for state-conditional HCA')
    parser.add_argument('-e', '--env', choices=['Shortcut', 'DelayedEffect', 'TabularPong'], required=True, metavar='ENV',
                        help='Can be Shortcut, DelayedEffect or TabularPong')

    agent_args = ['lr', 'h_lr', 'baseline', 'prior', 'temporal', 'discount', 'reward', 'h_dim', 'e_dim']
    agent_group = parser.add_argument_group(
        'Optional agent arguments')
    agent_group.add_argument('--lr', type=float,
                             help='Learning rate used for the actor, and the baseline if present')
    agent_group.add_argument('--h-lr', type=float,
                             help='Learning rate used for the hindsight estimator')
    agent_group.add_argument('--embed', action='store_const', const=True,
                             help='Flag to use embedding version of agent')
    agent_group.add_argument('--baseline', action='store_const', const=True,
                             help='Flag to use a baseline in PG')
    agent_group.add_argument('--factored', action='store_const', const=True,
                             help='Flag to use factored hindsight estimator in SHCA')
    agent_group.add_argument('--prior', action='store_const', const=True,
                             help='Flag to use the policy as prior in SHCA')
    agent_group.add_argument('--discount', type=float,
                             help='Discount rate for hindsight updates (not implemented for embeddings)')
    agent_group.add_argument('--reward', type=int,
                             help="""Method to handle immediate rewards. Can be:
    0: ignore
    1: r(S_t)
    2: r(S_t) * \pi(a|S_t)
    3: r(S_t) * h(a|S_t,S_t) / \pi(a|S_t)
    4: r*(S_t,a)
    5: r*(S_t,a) * \pi(a|S_t)
    6: r*(S_t,a) * h(a|S_t,S_t) / \pi(a|S_t)
    7: \hat r(S_t,a)
    8: \hat r(S_t,a) * \pi(a|S_t)
    9: \hat r(S_t,a) * h(a|S_t,S_t) / \pi(a|S_t)
    Defaults to 3 (not implemented for embeddings)""")
    agent_group.add_argument('--oracle', action='store_const', const=True,
                             help='Flag to use an SHCA agent with oracle hindsight distribution (not implemented for embeddings)')
    agent_group.add_argument('--temporal', action='store_const', const=True,
                             help='Flag to consider temporal distance in the oracle hindsight distribution')
    agent_group.add_argument('--variant', choices=['shared', 'separate', 'pair'], metavar='VAR',
                             help='Variant of embedding SHCA agent, can be "shared", "separate" or "pair"')
    agent_group.add_argument('--e-dim', type=int,
                             help='Size of state embeddings learned by embedding agents')
    agent_group.add_argument('--h-dim', type=int,
                             help='Size of the hidden layer in the hindsight estimator of EmbeddingSHCA agents')

    env_args = ['size', 'factors', 'actions', 'skip', 'final', 'noise', 'win']
    env_group = parser.add_argument_group('Optional environment arguments')
    env_group.add_argument('--size', type=int,
                           help='Length of the state chain in Shortcut and DelayedEffect, and the height and width of the grid in TabularPong')
    env_group.add_argument('--factors', type=int, nargs='*',
                           help='List of maximum values of noise factors corresponding to each dimension, defining their range starting from 0')
    env_group.add_argument('--actions', type=int,
                           help='Number of actions in each state for Shortcut and DelayedEffect, at most one action is optimal')
    env_group.add_argument('--skip', type=float,
                           help='Probability to skip to the final state with suboptimal actions in Shortcut')
    env_group.add_argument('--final', type=float,
                           help='Value of the final reward in Shortcut')
    env_group.add_argument('--noise', type=float,
                           help='Standard deviation for the reward noise distribution in DelayedEffect')
    env_group.add_argument('--win', type=float,
                           help='Probability for winning the game in TabularPong')

    train_args = ['eps', 'pre', 'add', 'prev']
    train_group = parser.add_argument_group('Optional training arguments')
    train_group.add_argument('--eps', type=int,
                             help='Number of episodes to train for')
    train_group.add_argument('--pre', type=int,
                             help='Number of episodes to pretrain for')
    train_group.add_argument('--add', type=int,
                             help='Number of aditional episodes to train hindsight for between each full agent updates')
    train_group.add_argument('--prev', type=int,
                             help='Number of past episodes to sample from for additional training. If 0 (default), then new episodes are sampled')

    args = parser.parse_args(sys.argv[1:])
    args.agent_cls = agent_cls(args.agent, args.embed, args.factored, args.variant, args.oracle)
    args.__dict__ = {k: v for k, v in vars(args).items() if v is not None}
    args.agent_args = {k: v for k, v in vars(args).items() if k in agent_args}
    args.env_args = {k: v for k, v in vars(args).items() if k in env_args}
    args.train_args = {k: v for k, v in vars(args).items() if k in train_args}
    return args
