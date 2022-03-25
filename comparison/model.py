from typing import Type
import torch
from torch import distributions as dist


# The following URL is a good resource on understanding the behaviour of
# `torch.Tensor` shapes with PyTorch distributions
#
# https://bochang.me/blog/posts/pytorch-distributions/


MEAN_SAMPLES = 100


class VAEForwardResult:
    def __init__(
        self,
        prior_dist: dist.Distribution,
        post_dist: dist.Distribution,
        lik_dist: dist.Distribution,
        xs: torch.Tensor,
        zs: torch.Tensor,
    ):
        self.prior_dist = prior_dist
        self.post_dist = post_dist
        self.lik_dist = lik_dist
        self.xs = xs
        self.zs = zs


class VAE(torch.nn.Module):
    """
    Base class for all implementations of VAE over datasets. 
    Encoding, decoding and distribution families need to be specified.
    """

    def __init__(
        self, post_class: Type[dist.Distribution], lik_class: Type[dist.Distribution],
    ):
        super().__init__()

        self.post_class = post_class
        self.lik_class = lik_class

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

    def prior_dist(self) -> dist.Distribution:
        """
        `p(z)` - the prior distribution over the latent space.
        """
        raise NotImplementedError

    def post_dist(self, xs: torch.Tensor) -> dist.Distribution:
        """
        `q(z | x)` - a variational approximation of the true posterior `p(z | x)` of the
        latent space given a sample point from the original space
        """
        post_params = self.encode(xs)
        return self.post_class(*post_params)

    def lik_dist(self, zs: torch.Tensor) -> dist.Distribution:
        """
        `p(x | z)` - the likelihood of reconstructing a sample point from the original space
        `x` given a point in the latent space `z`
        """
        lik_params = self.decode(zs)
        return self.lik_class(*lik_params)

    def forward(self, xs: torch.Tensor, K=1) -> dist.Distribution:
        """Given input `xs`, returns the function p(x|z)"""
        post_dist = self.post_dist(xs)
        zs = post_dist.rsample(torch.Size([K]))  # reparameterisation trick
        lik_dist = self.lik_dist(zs)
        return VAEForwardResult(
            prior_dist=self.prior_dist(),
            post_dist=post_dist,
            lik_dist=lik_dist,
            xs=xs,
            zs=zs,
        )

    @staticmethod
    def _mean(recon_dist) -> torch.Tensor:
        try:
            return recon_dist.mean.clone().detach()
        except NotImplementedError:
            return recon_dist.sample(torch.Size([MEAN_SAMPLES])).mean(dim=0)

    def reconstruct(self, xs: torch.Tensor) -> torch.Tensor:
        """Given input `xs`, approximate `xs` by mapping through latent space."""
        with torch.no_grad():
            zs = self.post_dist(xs).sample()
            recon_dist = self.lik_dist(zs)
            recon = self._mean(recon_dist)
        return recon

    def sample(self) -> torch.Tensor:
        """Samples random points `xs` from original space."""
        with torch.no_grad():
            zs = self.prior_dist().sample()
            recon_dist = self.lik_dist(zs)
            recon = self._mean(recon_dist)
        return recon

