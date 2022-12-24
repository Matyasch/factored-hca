import gym
import numpy as np


class DelayedEffect(gym.Env):
    """
    Delayed Effect environment from Hindsight Credit Assignment.
    """

    def __init__(self, size=5, actions=21, noise=2., factors=[]):
        """Initializes the environment

        Args:
            size (int, optional): Length of the state chain. Defaults to 5.
            actions (int, optional): Number of actions in each state. Defaults to 21.
            noise (float, optional): Standard deviation for the reward noise distribution. Defaults to 2.
            factors (list, optional): List of maximum values of noise factors corresponding to each dimension. Defaults to [].
        """
        super().__init__()
        self.size = size
        self.noise = noise
        self.factors = factors
        self.action_space = gym.spaces.Discrete(actions)
        self.observation_space = gym.spaces.MultiDiscrete([size + 1] + factors)
        self.observation_space.n = tuple(self.observation_space.nvec)  # This makes it compatible with agents
        self.optimal = np.random.choice(actions)

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
        if self.state == 0:  # Save initial action
            self.opt_chosen = action == self.optimal

        if self.state == self.size:  # Rewarding final state
            return -1, 1, True, {}
        elif self.state == self.size - 1:  # Penalizing final state
            return -1, -1, True, {}
        elif self.state == self.size - 2:  # Decide which final state to enter
            self.state = self.size if self.opt_chosen else self.size - 1
        else:  # Go to next state
            self.state += 1
        reward = np.random.normal(scale=self.noise)
        return self.states[self.state], reward, False, {}
