import gym
import numpy as np


class Shortcut(gym.Env):
    """
    Shortcut environment from Hindsight Credit Assignment.
    """

    def __init__(self, size=5, actions=21, skip=0.1, final=1., factors=[]):
        super().__init__()
        """Initializes the environment

        Args:
            size (int, optional): Length of the state chain. Defaults to 5.
            actions (int, optional): Number of actions in each state. Defaults to 21.
            skip (float, optional): Probability to skip to the final state with suboptimal actions. Defaults to 0.1.
            final (float, optional): Value of the final reward. Defaults to 1.
            factors (list, optional): List of maximum values of noise factors corresponding to each dimension. Defaults to [].
        """
        self.size = size
        self.skip = skip
        self.final = final
        self.factors = factors
        self.action_space = gym.spaces.Discrete(actions)
        self.observation_space = gym.spaces.MultiDiscrete([size] + factors)
        self.observation_space.n = tuple(self.observation_space.nvec)  # This makes it compatible with agents
        self.optimal = np.random.choice(actions, size=size - 1, replace=True)

    def reset(self):
        """Resets the environment and samples new noise factors

        Returns:
            tuple: observation vector (numpy array), info (dict)
        """
        self.states = np.stack([np.arange(self.size + 1)] +
                               [np.random.randint(num, size=self.size + 1) for num in self.factors], -1)
        self.state = 0
        return self.states[self.state], {}

    def step(self, action):
        """Takes an environment step based on the current state and the selected action

        Args:
            action (int): The selected action

        Returns:
            tuple: observation vector (numpy array), reward (int), terminated (bool), info (dict)
        """
        if self.state == self.size - 1:
            self.state = self.size
        elif action == self.optimal[self.state]:
            self.state = self.size - 1
        elif np.random.uniform() <= self.skip:
            self.state = self.size - 1
        else:
            self.state = self.state + 1

        done = self.state == self.size
        reward = self.final if done else -1

        return self.states[self.state], reward, done, {}
