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
        super().__init__(dist.normal.Normal, dist.bernoulli.Bernoulli)

        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc31 = nn.Linear(200, 50)  # mean of variational posterior q(z | x)
        self.fc32 = nn.Linear(200, 50)  # std of variational posterior q(z | x)
        self.fc4 = nn.Linear(50, 200)
        self.fc5 = nn.Linear(200, 200)
        self.fc6 = nn.Linear(200, 784)  #  mean of likelihood p(x | z)
        self.lik_std = torch.tensor(
            lik_std
        )  # std of likelihood p(x | z) is a hyperparameter

        self.device = 'cpu' # update device on forward pass

    def encode(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.device = xs.device
        h1 = torch.tanh(self.fc1(xs))
        h2 = torch.tanh(self.fc2(h1))
        mu = self.fc31(h2)
        std = torch.exp(self.fc32(h2))
        return mu, std

    def decode(self, zs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h3 = torch.tanh(self.fc4(zs))
        h4 = torch.tanh(self.fc5(h3))
        mu = torch.sigmoid(self.fc6(h3))
        return mu,

    def prior_dist(self) -> dist.Distribution:
        return dist.normal.Normal(
            torch.zeros(50, device=self.device), 
            torch.ones(50, device=self.device)
        )

    def encode_params(self):
        yield from self.fc1.parameters()
        yield from self.fc2.parameters()
        yield from self.fc31.parameters()
        yield from self.fc32.parameters()

    def decode_params(self):
        yield from self.fc4.parameters()
        yield from self.fc5.parameters()
        yield from self.fc6.parameters()

