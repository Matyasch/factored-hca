import numpy as np
import torch
import torch.nn as nn


def softmax(logits, dim=-1):
    """Numpy implementation of Sotmax

    Args:
        logits (numpy.array): Unnormalized logits

    Returns:
        numpy.array: Probability distributions
    """
    unnorm = np.exp(logits - logits.max(axis=dim, keepdims=True))
    return unnorm / unnorm.sum(axis=dim, keepdims=True)


def on_device(func):
    """Decorator that converts update function arguments to tensors and puts them on the device.

    Args:
        func: The update function.
    """

    def wrapper(self, states, actions, rewards, *args, **kwargs):
        """Convert arguments to tensors and put them on the device before passing them to the given update function.

        Args:
            states (numpy.array): Observed states.
            actions (numpy.array): Selected actions.
            rewards (numpy.array): Observed rewards.
        """
        states = torch.tensor(states).float().to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        func(self, states, actions, rewards, *args, **kwargs)
    return wrapper


class State2ID(nn.Linear):
    """Encodes state vectors into corresponding unique ids.

    Args:
        s_n (tuple): List of state factor ranges defined by their maximum value.
    """

    def __init__(self, s_n, **kwargs):
        super().__init__(len(s_n), 1, bias=False, **kwargs)
        multipliers = torch.cat((torch.tensor([1]), torch.cumprod(s_n[:-1], 0)), 0).float()
        self.weight = nn.Parameter(multipliers, requires_grad=False)

    def forward(self, states):
        """Encodes states into their corresponding IDs.

        Args:
            states (torch.Tensor): The state vectors to be encoded.

        Returns:
            torch.Tensor: The corresponding IDs.
        """
        return super().forward(states.float()).long()


class Hindsight(nn.Module):
    """Hindsight estimator module."""

    def __init__(self, a_n, e_dim=256, h_dim=128, **kwargs):
        """Initializes the hindsight estimator

        Args:
            a_n (int): Number of possible actions
            e_dim (int, optional): Size of state embeddings. Defaults to 256.
            h_dim (int, optional): Size of the hidden layer. Defaults to 128.
        """
        super().__init__(**kwargs)
        self.hindsight = nn.Sequential(
            nn.Linear(e_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, a_n)
        )

    def forward(self, state_pair_embeds):
        """Return the unnormalized hindsight distribution logits for input state pair embeddings.

        Args:
            state_pair_embeds (torch.Tensor): State pair embeddings.

        Returns:
            torch.Tensor: Unnormalized hindsight distribution logits.
        """
        return self.hindsight(state_pair_embeds)
