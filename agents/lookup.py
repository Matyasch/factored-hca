from functools import partial

import numpy as np

from agents.utils import softmax


class LookupAgent():
    """Parent class of lookup table agents."""

    def __init__(self, s_n, a_n, lr=0.4):
        """Initializes the lookup table agent parent class.

        Args:
            s_n (tuple): State factor ranges defined by their maximum value.
            a_n (int): Number of possible actions.
            lr (float, optional): Learning rate. Defaults to 0.4.
        """
        self.s_n = s_n
        self.a_n = a_n
        self.lr = lr
        self.actor = np.zeros(np.hstack((self.s_n, self.a_n)))  # lookup table for actor logits

    def forward(self, states):
        """Return unnormalized policy logits for input states.

        Args:
            state (numpy.array): Input states.

        Returns:
            numpy.array: Unnormalized policy logits.
        """
        return self.actor[tuple(states)]

    def act(self, state):
        """Sample an action according to the policy distribution for the input state.

        Args:
            state (numpy.array): The input state.

        Returns:
            int: Sampled action.
        """
        return np.random.choice(self.a_n, p=softmax(self.forward(state.T)))

    def update(self, states, actions, rewards, train_actor):
        """Update agent estimators according to input trajectory.

        Args:
            states (numpy.array): Observed states.
            actions (numpy.array): Selected actions.
            rewards (numpy.array): Observed rewards.
            train_actor (bool): Whether to train the actor.

        Raises:
            NotImplementedError: To be implemented by agents.
        """
        raise NotImplementedError


class LookupPG(LookupAgent):
    """Policy Gradient baseline using learnable lookup tables."""

    def __init__(self, baseline=False, **kwargs):
        """Initializes the LookupPG agent.

        Args:
            baseline (bool, optional): Whether to use a baseline. Defaults to False.
        """
        super().__init__(**kwargs)
        self.use_baseline = baseline
        self.baseline = np.zeros((self.s_n))  # lookup table for state values

    def update(self, states, actions, rewards, **kwargs):
        """Update the actor, and the baseline if present, using REINFORCE

        Args:
            states (numpy.array): Observed states.
            actions (numpy.array): Selected actions.
            rewards (numpy.array): Observed rewards.
        """
        states, rewards, actions = tuple(states.T), rewards, (actions,)
        returns = rewards + rewards.sum() - rewards.cumsum(0) - self.baseline[states]
        np.add.at(self.actor, (states + actions), returns * self.lr)
        np.add.at(self.actor, states, -softmax(self.forward(states)) * np.expand_dims(returns, 1) * self.lr)
        if self.use_baseline:
            np.add.at(self.baseline, states, returns * self.lr)


class LookupSHCA(LookupAgent):
    """State-conditional HCA Agent using learnable lookup tables."""

    def __init__(self, factored=False, h_lr=0.4, prior=False, discount=1, reward=3, **kwargs):
        """Initializes the LookupSHCA agent.

        Args:
            h_lr (float, optional): Learning rate of the hindsight estimator. Defaults to 0.4.
            prior (bool, optional): Whether to use the policy as prior. Defaults to False.
            discount (int, optional): Temporal discount rate in hindsight estimator updates. Defaults to 1.
            reward (int, optional): Different methods to handle immediate reward:
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
                Defaults to 3.
        """
        super().__init__(**kwargs)
        self.factored = factored
        self.h_lr = h_lr
        self.prior = prior
        self.reward = reward
        self.discount = discount

        # Lookup table for hindsight logits
        self._hindsight = np.zeros(np.hstack((self.s_n, self.s_n, self.a_n)))

        # Select mode
        if reward in [1, 2, 3]:  # Observed
            self.reward = lambda s, r: r
        elif reward in [4, 5, 6]:  # Oracle: observed in Shortcut and Tabular Pong (deterministic), 0 in Delayed Effect
            self.reward = lambda s, r: r if np.all(r[:-1] == -1) or np.all(r[:-1] == 0) else 0
        elif reward in [7, 8, 9]:  # Estimated
            self._reward = np.zeros(np.hstack((self.s_n, self.a_n)))
            self.reward = lambda s, r: self._reward[s].copy()
        else:  # Ignore
            self.reward = lambda s, r: 0

        if reward in [1, 4, 7]:  # Do not weight
            self.weight = lambda s: 1
        elif reward in [2, 5, 8]:  # Weight by \pi(a|S_t)
            self.weight = lambda s: softmax(self.forward(s))
        elif reward in [3, 5, 9]:  # Weight by h(a|S_t,S_t) / \pi(a|S_t)
            self.weight = lambda s: softmax(self.hindsight(s, s)) - softmax(self.forward(s))
        else:  # Ignore:
            self.weight = lambda s: 0

    def hindsight(self, states, futures):
        """Computes unnormalized state-conditional hindsight logits.

        Args:
            states (tuple): Initial states.
            futures (tuple): Future states.

        Returns:
            numpy.array: Unnormalized hindsight logits.
        """
        return self._hindsight[states + futures] + (np.log(softmax(self.forward(states))) if self.prior else 0)

    def update_hindsight(self, states, futures, actions, discount):
        """Updates the hindsight estimator using cross entropy loss

        Args:
            states (tuple): Initial states.
            futures (tuple): Future states.
            actions (tuple): Selected actions in initial states.
            discount (list): Discount rates for each reward.
        """
        h_prob = softmax(self.hindsight(states, futures))
        np.add.at(self._hindsight, states + futures + actions, discount * self.h_lr)
        np.add.at(self._hindsight, states + futures, -h_prob * np.expand_dims(discount, 1) * self.h_lr)

    def update(self, states, actions, rewards, train_actor=True):
        """Update agent estimators using hindisght credit assignment.

        Args:
            states (numpy.array): Observed states.
            actions (numpy.array): Selected actions.
            rewards (numpy.array): Observed rewards.
            train_actor (bool, optional): Whether to train the actor. Defaults to True.
        """
        t, k = np.triu_indices(len(actions))  # time step pairs
        # Index initial and future time steps, and convert to tuples for lookup table indexing
        s_t, a_t, s_k, r_k = tuple(states[t].T), (actions[t],), tuple(states[k].T), np.expand_dims(rewards[k], 1)
        states, actions = tuple(states.T), (actions,)  # convert to tuples for lookup table indexing
        discount = self.discount**(k - t)
        discount[k == t + 1] = 1

        self.update_hindsight(s_t, s_k, a_t, discount)
        if self.reward in [7, 8, 9]:  # Update reward estimator
            np.add.at(self._reward, states + actions, (rewards - self._reward[states + actions]) * self.lr)
        if train_actor:
            weighted_r_k = r_k * (softmax(self.hindsight(s_t, s_k)) - softmax(self.forward(s_t)))
            weighted_r_k[t == k] = self.reward(states, np.expand_dims(rewards, 1)) * self.weight(states)
            returns = np.zeros((len(states[0]), self.a_n))
            np.add.at(returns, t, weighted_r_k)
            # Update actor
            np.add.at(self.actor, states, returns * self.lr)
            np.add.at(self.actor, states, -softmax(self.forward(states)) * np.expand_dims(returns.sum(1), 1) * self.lr)


class FactoredLookupSHCA(LookupSHCA):
    """State-conditional HCA Agent using learnable lookup tables and factored hindsight distribution."""

    def __init__(self, **kwargs):
        """Initializes the factored LookupSHCA agent."""
        super().__init__(**kwargs)
        self._hindsight = np.zeros(np.hstack((self.s_n[0], self.s_n[0], self.a_n)))

    def hindsight(self, states, futures):
        """Computes unnormalized state-conditional hindsight logits.

        Args:
            states (tuple): Initial states.
            futures (tuple): Future states.

        Returns:
            numpy.array: Unnormalized hindsight logits.
        """
        return self._hindsight[(states[0], futures[0])] + (np.log(softmax(self.forward(states))) if self.prior else 0)

    def update_hindsight(self, states, futures, actions, discount):
        """Updates the hindsight estimator using cross entropy loss

        Args:
            states (tuple): Initial states.
            futures (tuple): Future states.
            actions (tuple): Selected actions in initial states
            discount (list): Discount rates for each reward.
        """
        h_prob = softmax(self.hindsight(states, futures))
        states, futures = (states[0],), (futures[0],)
        np.add.at(self._hindsight, states + futures + actions, discount * self.h_lr)
        np.add.at(self._hindsight, states + futures, -h_prob * np.expand_dims(discount, 1) * self.h_lr)


class OracleSHCA(LookupSHCA):
    """State-conditional HCA Agent using oracle hindsight distribution."""

    def __init__(self, env, temporal=False, **kwargs):
        """Initializes the OracleSHCA agent.

        Args:
            env (gym.Env): The environment instance
            temporal (bool, optional): Whether to consider temporal distance on Shortcut. Defaults to False.

        Raises:
            NotImplementedError: Only implemented for Shortcut and Delayed Effect
        """
        super().__init__(**kwargs)
        if env.spec.id.split('-')[0] == 'Shortcut':
            self.hindsight = partial(self.hindsight_shortcut, env=env, temporal=temporal)
        elif env.spec.id.split('-')[0] == 'DelayedEffect':
            self.hindsight = partial(self.hindsight_delayedeffect, env=env)
        else:
            raise NotImplementedError('Only implemented for Shortcut and Delayed Effect')

    def update_hindsight(self, *args, **kwargs):
        """Do not update oracle hindsight."""
        pass

    def hindsight_shortcut(self, states, futures, env, temporal=False):
        """Computes state-conditional hindsight logits for a specific Shortcut instance.

        Args:
            states (tuple): Initial states.
            futures (tuple): Future states.
            env (gym.Env): Shortcut environment instance
            temporal (bool, optional): Whether to consider temporal distance. Defaults to False.

        Returns:
            numpy.array: Hindsight logits
        """
        s_full, s_true, f_true = np.array(states).T, states[0], futures[0]  # oracle knows the relevant factor
        inds = np.where(s_true != f_true)[0]  # state pairs where s_t != s_k
        probs = softmax(self.forward(states))  # inizialize according to policy

        for i, s_n, s_t, f_t in zip(inds, s_full[inds], s_true[inds], f_true[inds]):
            subopt = np.arange(self.a_n)[np.arange(self.a_n) != env.optimal[s_t]]  # suboptimal actions
            s_n = tuple(s_n)
            # Future intermediate state
            if f_t < env.size - 1:
                probs[i, env.optimal[s_t]] = 0  # optimal action does not lead to intermediate state
                probs[i, subopt] = softmax(self.forward(s_n)[subopt])  # policy distribution over subopt actions
            # Future final state: hindsight distribution is identical to policy if temporal distance is not considered
            elif temporal:
                probs[i, subopt] = 0  # Ignore suboptimal actions
                probs[i, env.optimal[s_t]] = 1  # Assign all probability to optimal action
        return np.log(probs)

    def hindsight_delayedeffect(self, states, futures, env):
        """Computes state-conditional hindsight logits for a specific Delayed Effect instance.

        Args:
            states (tuple): Initial states.
            futures (tuple): Future states.
            env (gym.Env): Delayed Effect environment instance

        Returns:
            numpy.array: Hindsight logits
        """
        probs = softmax(self.forward(states))  # inizialize according to policy
        subopt = np.arange(self.a_n)[np.arange(self.a_n) != env.optimal]  # suboptimal actions
        ind = env.size - 1  # Index of (S_0, S_T) pair
        probs[ind] = 0
        if futures[0][-1] == env.size:  # Rewarding final state
            probs[ind, env.optimal] = 1
        else:  # Penalizing case
            probs[ind, subopt] = softmax(self.forward(np.array(states)[:, ind])[subopt])
        return np.log(probs)
