import numpy as np
import torch
import torch.nn as nn
from network_utils import build_mlp, device, np2torch


class BaselineNetwork(nn.Module):
    """
    Class for implementing Baseline network
    """

    def __init__(self, state_dim, lr, n_layers, layer_size):
        super().__init__()
        self.baseline = None
        self.lr = 0.001
        observation_dim = state_dim
        self.network = build_mlp(observation_dim, 1, n_layers, layer_size)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def forward(self, observations):
        """
        Args:
            observations: torch.Tensor of shape [batch size, dim(observation space)]
        Returns:
            output: torch.Tensor of shape [batch size]
        """
        output = self.network(observations).squeeze()
        return output

    def calculate_advantage(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size]
                all discounted future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        Returns:
            advantages: np.array of shape [batch size]
        """
        observations = np2torch(observations)
        baseline = self(observations).detach().numpy()
        advantages = returns - baseline
        return advantages

    def update_baseline(self, returns, observations):
        """
        Args:
            returns: np.array of shape [batch size], containing all discounted
                future returns for each step
            observations: np.array of shape [batch size, dim(observation space)]
        """
        returns = np2torch(returns)
        observations = np2torch(observations)
        baseline = self(observations)
        loss = nn.functional.mse_loss(baseline, returns)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
