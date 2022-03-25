import torch

from comparison.model import VAEForwardResult

# TODO - Fix shapes


def ELBO(res: VAEForwardResult) -> torch.Tensor:
    return (
        res.prior_dist.log_prob(res.zs)
        + res.lik_dist.log_prob(res.xs)
        - res.post_dist.log_prob(res.zs)
    ).mean(dim=0)


def IWAE(res: VAEForwardResult) -> torch.Tensor:
    return (
        res.prior_dist.log_prob(res.zs)
        + res.lik_dist.log_prob(res.xs)
        - res.post_dist.log_prob(res.zs)
    )
