from collections import deque
from random import sample

import numpy as np
from tqdm import trange


class Trainer:
    """Class to handle training"""

    def __init__(self, eps=500, pre=0, add=0, prev=0):
        """Initialize trainer class

        Args:
            eps (int, optional): Number of episodes to train for. Defaults to 500.
            pre (int, optional): Number of episodes to pretrain for. Defaults to 0.
            add (int, optional): Number of additional episodes to train hindsight for between each full agent updates. Defaults to 0.
            prev (int, optional): Number of past episodes to sample from for additional training. If 0, then new episodes are sampled. Defaults to 0.
        """
        self.eps = eps
        self.pre = pre
        self.add = add
        self.prev_episodes = deque(maxlen=prev)

    def run_episode(self, agent, env) -> tuple:
        """Run an episode using the given agent in the given environment

        Args:
            agent: Agent to run episode with
            env: Environment to run episode in

        Returns:
            tuple: Observation vectors (numpy array), selected actions (list), rewards (list)
        """
        states = []
        actions = []
        rewards = []
        done = False
        s, _ = env.reset()
        while not done:
            a = agent.act(s)
            ns, r, done, _ = env.step(a)
            states.append(s)
            actions.append(a)
            rewards.append(r)
            s = ns
        return np.array(states), np.array(actions), np.array(rewards)

    def additional_episodes(self, agent, env) -> list:
        """Sample episodes for the additional training of the hindsight estimator

        Args:
            agent: Agent to use if new episodes are sampled
            env: Environment to use if new episodes are sampled

        Returns:
            list: Sampled episodes
        """
        if len(self.prev_episodes) >= self.add:
            return sample(self.prev_episodes, k=self.add)
        elif not self.prev_episodes:
            return [self.run_episode(agent, env) for _ in range(self.add)]
        else:
            return []

    def fit(self, agent, env) -> list:
        """Train the given agent on the given environment

        Args:
            agent: Agent to train
            env: Environment to train on

        Returns:
            list: Achieved returns
        """
        returns = []
        # Pretrain hindsight
        for _ in range(self.pre):
            states, actions, rewards = self.run_episode(agent, env)
            agent.update(states, actions, rewards, train_actor=False)
        # Train actor and hindsight
        for ep in trange(self.eps, leave=False):
            states, actions, rewards = self.run_episode(agent, env)
            agent.update(states, actions, rewards)
            returns.append(rewards[-1] if env.spec.id == 'DelayedEffect-v0' else sum(rewards))
            # Additional training for hindsight
            self.prev_episodes.append((states, actions, rewards))
            for states, actions, rewards in self.additional_episodes(agent, env):
                agent.update(states, actions, rewards, train_actor=False)
        return returns
