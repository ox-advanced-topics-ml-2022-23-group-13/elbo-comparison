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

    def __init__(self):
        super().__init__(dist.normal.Normal, dist.normal.Normal, dist.normal.Normal)

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, xs):
        # estimates mu, std of approximate posterior q(z|x)
        h1 = F.relu(self.fc1(xs))
        return self.fc21(h1), torch.exp(self.fc22(h1))

    def decode(self, zs):
        h3 = F.relu(self.fc3(zs))
        return torch.sigmoid(self.fc4(h3)), torch.tensor(0.1).to(zs.device)

