import torch
from torch import nn
import torch.nn.functional as F
import torch.distributions as dist

from comparison.model import VAE


class VAE_Toy(VAE):
    """
    An implementation of the toy model from the paper.
    """

    def __init__(self, D):
        super().__init__(
            dist.normal.Normal, 
            dist.normal.Normal
        )

        self.D = D

        self.inf_layer = nn.Linear(D, D)
        self.mu = nn.Parameter(torch.rand(self.D), requires_grad=True)

        self.device = 'cpu' # update device on forward pass

    def encode(self, xs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.device = xs.device
        return self.inf_layer(xs), torch.ones(self.D, device=xs.device) * .6667

    def decode(self, zs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return zs, torch.ones(self.D, device=zs.device)

    def prior_dist(self) -> dist.Distribution:
        return dist.normal.Normal(
            torch.zeros(self.mu, device=self.device), 
            torch.ones(self.D, device=self.device)
        )

