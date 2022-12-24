import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.utils import on_device, State2ID, Hindsight


class EmbeddingAgent(nn.Module):
    """Parent class of embedding agents."""

    def __init__(self, s_n, a_n, eps, lr=1e-3, e_dim=256):
        """Initializes the embedding agent parent class.

        Args:
            s_n (tuple): State factor ranges defined by their maximum value.
            a_n (int): Number of possible actions.
            eps (int): Number of episodes to train for.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            e_dim (int, optional): Size of state embeddings. Defaults to 256.
        """
        super().__init__()
        self.factors = torch.tensor(s_n)
        self.s_n = self.factors.prod()
        self.a_n = a_n
        self.lr = lr
        self.e_dim = e_dim

        self.s2i = State2ID(self.factors)
        self.embed = nn.Embedding(self.s_n, e_dim)
        self.actor = nn.Sequential(self.s2i, self.embed, nn.Linear(e_dim, self.a_n))

        self.optim = torch.optim.Adam(self.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=eps, eta_min=lr * 0.1)

        self.device = torch.device("cpu")
        self.to(self.device)

    def forward(self, state):
        """Return unnormalized policy logits for input states.

        Args:
            state (torch.Tensor): Input states.

        Returns:
            torch.Tensor: Unnormalized policy logits.
        """
        return self.actor(state)

    def act(self, state):
        """Sample an action according to the policy distribution for the input state.

        Args:
            state (numpy.array): The input state.

        Returns:
            int: Sampled action.
        """
        state = torch.tensor(state).to(self.device)
        a = Categorical(logits=self(state)).sample()
        return a.item()

    @on_device
    def update(self, states, actions, rewards, train_actor):
        """Update agent estimators according to input trajectory.

        Args:
            states (torch.Tensor): Observed states.
            actions (torch.Tensor): Selected actions.
            rewards (torch.Tensor): Observed rewards.
            train_actor (bool): Whether to train the actor.

        Raises:
            NotImplementedError: To be implemented by agents.
        """
        raise NotImplementedError


class EmbeddingPG(EmbeddingAgent):
    """Policy Gradient baseline using learnable state embedding representations."""

    def __init__(self, baseline=False, **kwargs):
        """Initializes the EmbeddingPG agent

        Args:
            baseline (bool, optional): Whether to use a baseline. Defaults to False.
        """
        super().__init__(**kwargs)
        self.use_baseline = baseline
        if self.use_baseline:
            self._baseline = nn.Linear(self.e_dim, 1)
            self.baseline = nn.Sequential(self.s2i, self.embed, self._baseline)
            self.optim.add_param_group({'params': self._baseline.parameters()})
        self.to(self.device)

    @on_device
    def update(self, states, actions, rewards, **kwargs):
        """Update the actor, and the baseline if present, using REINFORCE

        Args:
            states (torch.Tensor): Observed states.
            actions (torch.Tensor): Selected actions.
            rewards (torch.Tensor): Observed rewards.
        """
        returns = rewards + rewards.sum() - rewards.cumsum(0)  # fast reverse cumsum

        baseline_loss = 0
        if self.use_baseline:
            baseline = self.baseline(states).squeeze()
            returns -= baseline.detach()
            baseline_loss = F.mse_loss(baseline, returns)

        log_probs = Categorical(logits=self(states)).log_prob(actions)
        actor_loss = -(log_probs * returns.detach()).mean()

        self.optim.zero_grad()
        (actor_loss + baseline_loss).backward()
        self.optim.step()
        self.scheduler.step()


class EmbeddingSHCA(EmbeddingAgent):
    """State-conditional HCA Agent using learnable state embedding representations."""

    def __init__(self, h_lr=1e-3, prior=False, h_dim=128, **kwargs):
        """Initializes the EmbeddingSHCA agent.

        Args:
            h_lr (float, optional): Learning rate of the hindsight estimator. Defaults to 1e-3.
            prior (bool, optional): Whether to use the policy as prior. Defaults to False.
            h_dim (int, optional): Size of the hidden layer in the hindsight estimator. Defaults to 128.
        """
        super().__init__(**kwargs)
        self.h_lr = h_lr
        self.prior = prior
        self.h_dim = h_dim
        self.embed_h, self._hindsight = self.components()
        self.h_optim = torch.optim.Adam([
            {'params': self.embed_h.parameters()},
            {'params': self._hindsight.parameters()}
        ], lr=self.h_lr)
        self.to(self.device)

    def hindsight(self, states, futures):
        """Computes unnormalized state-conditional hindsight logits.

        Args:
            states (torch.Tensor): Initial states.
            futures (torch.Tensor): Future states.

        Returns:
            torch.Tensor: Unnormalized hindsight logits.
        """
        pair_embeds = self.embed_state_pairs(states, futures)
        return self._hindsight(pair_embeds) + (F.log_softmax(self.actor(states)).detach() if self.prior else 0)

    def update_hindsight(self, states, futures, actions):
        """Updates the hindsight estimator using cross entropy loss

        Args:
            states (torch.Tensor): Initial states.
            futures (torch.Tensor): Future states.
            actions (torch.Tensor): Selected actions in initial states
        """
        hindsight_loss = F.cross_entropy(self.hindsight(states, futures), actions)
        self.h_optim.zero_grad()
        hindsight_loss.backward()
        self.h_optim.step()

    @on_device
    def update(self, states, actions, rewards, train_actor=True):
        """Update agent estimators using hindisght credit assignment.

        Args:
            states (numpy.array): Observed states.
            actions (numpy.array): Selected actions.
            rewards (numpy.array): Observed rewards.
            train_actor (bool, optional): Whether to train the actor. Defaults to True.
        """
        t, k = torch.triu_indices(len(actions), len(actions))  # time step pairs
        s_t, a_t, s_k, r_k = states[t], actions[t], states[k], rewards[k]
        self.update_hindsight(s_t, s_k, a_t)
        if train_actor:
            weighted_r_k = r_k.unsqueeze(1) * (F.softmax(self.hindsight(s_t, s_k), 1) / F.softmax(self.actor(s_t), 1))
            returns = torch.zeros(len(rewards), self.a_n)
            returns = returns.index_add(0, t, weighted_r_k).detach()
            # Update actor
            actor_loss = -(F.softmax(self.actor(states), 1) * returns).sum()
            self.optim.zero_grad()
            actor_loss.backward()
            self.optim.step()
            self.scheduler.step()

    def components(self):
        """Initialize hindsight embedding and hindsight estimator.

        Raises:
            NotImplementedError: To be implemented by variants.
        """
        raise NotImplementedError('To be implemented by variants.')

    def embed_state_pairs(self, states, futures):
        """Compute state-pair embeddings.

        Args:
            states (torch.Tensor): Initial states.
            futures (torch.Tensor): Future states.

        Raises:
            NotImplementedError: To be implemented by variants.
        """
        raise NotImplementedError('To be implemented by variants.')


class SeparateSHCA(EmbeddingSHCA):
    """EmbeddingSHCA agent that concatenates embeddings of individual states,
    obtained using a separate embedding layer."""

    def components(self):
        """Initialize hindsight embedding as a separate embedding layer
        and the hindsight estimator with an input twice the emedding size.

        Returns:
            tuple: Hindsight emeddings and estimator modules.
        """
        embed_h = nn.Embedding(self.s_n, self.e_dim)
        hindsight = Hindsight(self.a_n, 2 * self.e_dim, self.h_dim)
        return embed_h, hindsight

    def embed_state_pairs(self, states, futures):
        """Embed state vectors and concatenate corresponding pairs.

        Args:
            states (torch.Tensor): Initial states.
            futures (torch.Tensor): Future states.

        Returns:
            torch.Tensor: State par embeddings.
        """
        return torch.cat((self.embed_h(self.s2i(states)), self.embed_h(self.s2i(futures))), 1)


class SharedSHCA(SeparateSHCA):
    """EmbeddingSHCA agent that concatenates embeddings of individual states,
    obtained using the same embedding layer as the actor."""

    def components(self):
        """Initialize hindsight embedding to be the same as the actor embedding layer
        and the hindsight estimator with an input twice the emedding size.

        Returns:
            tuple: Hindsight emeddings and estimator modules.
        """
        embed_h = self.embed
        hindsight = Hindsight(self.a_n, 2 * self.e_dim, self.h_dim)
        return embed_h, hindsight


class PairSHCA(EmbeddingSHCA):
    """EmbeddingSHCA agent that embeds state pairs,
    obtained using a separate embedding layer."""

    def components(self):
        """Initialize hindsight embedding such that it has a different embedding for each state pair
        and the hindsight estimator with an input of the emedding size.

        Returns:
            tuple: Hindsight emeddings and estimator modules.
        """
        embed_h = nn.Embedding(self.s_n**2, self.e_dim)
        hindsight = Hindsight(self.a_n, self.e_dim, self.h_dim)
        return embed_h, hindsight

    def embed_state_pairs(self, states, futures):
        """Embed state vector pairs.

        Args:
            states (torch.Tensor): Initial states.
            futures (torch.Tensor): Future states.

        Returns:
            torch.Tensor: State par embeddings.
        """
        pair_inds = self.s2i(states) * self.s_n + self.s2i(futures)
        return self.embed_h(pair_inds)


class FactoredPairSHCA(PairSHCA):
    """EmbeddingSHCA agent that embeds only the relevant factors of state pairs,
    obtained using a separate embedding layer."""

    def components(self):
        """Initialize hindsight embedding such that it has a different embedding for each relevant factor pair
        and the hindsight estimator with an input of the emedding size.

        Returns:
            tuple: Hindsight emeddings and estimator modules.
        """
        embed_h = nn.Embedding(self.factors[0]**2, self.e_dim)
        hindsight = Hindsight(self.a_n, self.e_dim, self.h_dim)
        return embed_h, hindsight

    def embed_state_pairs(self, states, futures):
        """Embed relevant factors of state pairs.

        Args:
            states (torch.Tensor): Initial states.
            futures (torch.Tensor): Future states.

        Returns:
            torch.Tensor: State par embeddings.
        """
        pair_inds = states[:, 0] * self.factors[0] + futures[:, 0]
        return self.embed_h(pair_inds.long())
