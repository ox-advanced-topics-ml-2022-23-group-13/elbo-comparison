import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

from comparison.model import VAE


class VAE_MNIST(VAE):
    """
    A reimplementation of code from ATML Practical 3
    """

    def __init__(self, lik_std=0.1):
        super().__init__(dist.normal.Normal, dist.normal.Normal)

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)  # mean of variational posterior q(z | x)
        self.fc22 = nn.Linear(400, 20)  # std of variational posterior q(z | x)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)  #  mean of likelihood p(x | z)
        self.lik_std = torch.tensor(
            lik_std
        )  # std of likelihood p(x | z) is a hyperparameter

        self.device = 'cpu' # update device on forward pass

    def encode(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.device = xs.device
        h1 = F.relu(self.fc1(xs))
        mu = self.fc21(h1)
        std = torch.exp(self.fc22(h1))
        return mu, std

    def decode(self, zs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h3 = F.relu(self.fc3(zs))
        mu = torch.sigmoid(self.fc4(h3))
        std = torch.ones(784, device=zs.device) * self.lik_std.to(zs.device)
        return mu, std

    def prior_dist(self) -> dist.Distribution:
        return dist.normal.Normal(
            torch.zeros(20, device=self.device), 
            torch.ones(20, device=self.device)
        )

    def encode_params(self):
        yield from self.fc1.parameters()
        yield from self.fc21.parameters()
        yield from self.fc22.parameters()

    def decode_params(self):
        yield from self.fc3.parameters()
        yield from self.fc4.parameters()

