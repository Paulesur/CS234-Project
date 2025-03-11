import torch
import torch.nn as nn
import torch.distributions as ptd
import numpy as np

from network_utils import np2torch, device


class BasePolicy:
    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: instance of a subclass of torch.distributions.Distribution
        """
        raise NotImplementedError

    def act(
        self,
        observations,
    ):
        """
        Args:
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            sampled_actions: np.array of shape [batch size, *shape of action]
            log_probs: np.array of shape [batch size] (optionally, if return_log_prob)
        """
        observations = np2torch(observations)
        distribution = self.action_distribution(observations)
        sampled_actions = distribution.sample()
        log_probs = distribution.log_prob(sampled_actions).detach().numpy()
        sampled_actions = sampled_actions.detach().numpy()
        return sampled_actions, log_probs


class CategoricalPolicy(BasePolicy, nn.Module):
    def __init__(self, network):
        nn.Module.__init__(self)
        self.network = network

    def action_distribution(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            distribution: torch.distributions.Categorical where the logits
                are computed by self.network
        """
        logits = self.network(observations)
        distribution = ptd.Categorical(logits=logits)
        return distribution

    def forward(self, x):
        return self.network(x)
