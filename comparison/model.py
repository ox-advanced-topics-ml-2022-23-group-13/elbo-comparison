from typing import Type
import torch
from torch import distributions as dist


MEAN_SAMPLES = 100


class VAE(torch.nn.Module):
    """
    Base class for all implementations of VAE over datasets. 
    Encoding, decoding and distribution families need to be specified.
    """
    
    def __init__(
        self,
        prior_dist: Type[dist.Distribution],
        post_dist: Type[dist.Distribution],
        lik_dist: Type[dist.Distribution],
    ):
        super().__init__()

        self.prior_dist = prior_dist
        self.post_dist = post_dist
        self.lik_dist = lik_dist

    def encode(self, xs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Maps original distribution data to parameters of distributions over latent space.

        Input Shape: `(B, *original.shape)`,
        Output: `((B, *param.shape) for param in post_params)`
        """
        raise NotImplementedError

    def decode(self, zs) -> tuple[torch.Tensor, ...]:
        """
        Maps latent space values to parameters of posterior over original distribution.
        
        Input Shape: `(B, *latent_shape)`,
        Output: `((B, *param.shape) for param in lik_params)`
        """
        raise NotImplementedError

    def forward(self, xs: torch.Tensor) -> dist.Distribution:
        """Given input `xs`, returns the function p(x|z)"""
        post_params = self.encode(xs)
        zs = self.post_dist(*post_params).rsample()  # reparameterisation trick
        lik_params = self.decode(zs)
        return self.lik_dist(*lik_params)

    def reconstruct(self, xs: torch.Tensor) -> torch.Tensor:
        """Given input `xs`, approximate `xs` by mapping through latent space."""
        with torch.no_grad():
            post_dist = self.forward(xs)
            mean = post_dist.sample(torch.Size([MEAN_SAMPLES])).mean(dim=0)
        return mean
