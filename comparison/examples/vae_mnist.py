import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

from comparison.model import VAE


class VAE_MNIST(VAE):
    """
    A reimplementation of code from ATML Practical 3
    """

    def __init__(self, lik_std=0.1):
        super().__init__(dist.multivariate_normal.MultivariateNormal, dist.multivariate_normal.MultivariateNormal)

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # mean of variational posterior q(z | x)
        self.fc22 = nn.Linear(400, 20)  # std of variational posterior q(z | x)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)  #  mean of likelihood p(x | z)
        self.lik_std = torch.tensor(
            lik_std
        )  # std of likelihood p(x | z) is a 

        self.device = 'cpu' # update device on forward pass

    def encode(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.device = xs.device
        h1 = F.relu(self.fc1(xs))
        return self.fc21(h1), torch.diag_embed(torch.exp(self.fc22(h1)))

    def decode(self, zs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h3 = F.relu(self.fc3(zs))
        std = torch.eye(n = 784, device=zs.device) * self.lik_std.to(zs.device)
        return torch.sigmoid(self.fc4(h3)), std

    def prior_dist(self) -> dist.Distribution:
        return dist.multivariate_normal.MultivariateNormal(
            torch.zeros(20, device=self.device), 
            torch.eye(20, device=self.device)
        )

