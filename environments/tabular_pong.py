import gym
import numpy as np


class TabularPong(gym.Env):
    '''
    Tabular Pong environment
    '''

    def __init__(self, size=4, win=0.1, factors=[]):
        """Initializes the environment

        Args:
            size (int, optional): Height and width of the grid. Defaults to 4.
            win (float, optional): Probability for winning the game. Defaults to 0.1.
            factors (list, optional): List of maximum values of noise factors corresponding to each dimension, defining their range starting from 0. Defaults to [].
        """
        super().__init__()
        self.size = size - 1
        self.win = win
        self.factors = factors
        self.action_space = gym.spaces.Discrete(3)

        self.i2s = []
        self.s2i = {}
        for py in range(self.size + 1):
            for by in range(self.size + 1):
                for bx in range(-1, self.size + 2):
                    for bvy in [-1, 0, 1]:
                        for bvx in [-1, 1]:
                            self.i2s.append((py, by, bx, bvy, bvx))
                            self.s2i[(py, by, bx, bvy, bvx)] = len(self.s2i)
        self.observation_space = gym.spaces.MultiDiscrete([len(self.i2s)] + factors)
        self.observation_space.n = tuple(self.observation_space.nvec)  # This makes it compatible with agents

    def reset(self) -> tuple:
        """Resets the environment and samples new noise factors

        Returns:
            tuple: observation vector (numpy array), info (dict)
        """
        self.states = np.stack([np.arange(len(self.i2s))] +
                               [np.random.randint(num, size=len(self.i2s)) for num in self.factors], -1)

        self.py = self.size // 2
        self.by = self.size // 2
        self.bx = self.size // 2
        self.bvy = np.random.choice([-1, 0, 1])
        self.bvx = np.random.choice([-1, 1])

        return self.states[self.s2i[(self.py, self.by, self.bx, self.bvy, self.bvx)]], {}

    def step(self, action: int) -> tuple:
        """Takes an environment step based on the current state and the selected action

        Args:
            action (int): The selected action

        Returns:
            tuple: observation vector (numpy array), reward (int), terminated (bool), info (dict)
        """
        r = 0
        done = False

        # Copy state
        py = self.py
        by = self.by
        bx = self.bx
        bvy = self.bvy
        bvx = self.bvx

        # Bouncing on X axis
        if bx + bvx < 0:  # Ball at player's end
            if py == by:  # Player touches the ball
                bvy = np.random.choice([-1, 0, 1])  # New random Y direction on bounce
                bvx = -bvx
            else:  # Player misses the ball and loses
                r = -1
                done = True
        elif bx + bvx > self.size:  # Ball at opposite wall
            if np.random.uniform() < self.win:  # Player win
                r = 1
                done = True
            else:  # Ball bounces back
                bvy = np.random.choice([-1, 0, 1])  # New random Y direction on bounce
                bvx = -bvx

        # Bouncing on Y axis
        if not 0 <= by + bvy <= self.size:
            bvy = -bvy

        # Progress to next state
        py = np.clip(py + action - 1, 0, self.size)
        by = by + bvy
        bx = bx + bvx

        # Update state
        self.py = py
        self.by = by
        self.bx = bx
        self.bvy = bvy
        self.bvx = bvx

        return self.states[self.s2i[(py, by, bx, bvy, bvx)]], r, done, {}
