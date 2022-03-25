import torch

from comparison.model import VAEForwardResult


SAMPLE_DIM = 0
BATCH_DIM = 1


def ELBO(res: VAEForwardResult) -> torch.Tensor:
    return (
        res.prior_dist.log_prob(res.zs)
        + res.lik_dist.log_prob(res.xs)
        - res.post_dist.log_prob(res.zs)
    ).mean(dim=BATCH_DIM).mean(dim=SAMPLE_DIM)


def IWAE(res: VAEForwardResult) -> torch.Tensor:
    K = torch.tensor(res.zs.size(SAMPLE_DIM), device=res.xs.device)
    return (
        torch.logsumexp(
            res.prior_dist.log_prob(res.zs)
            + res.lik_dist.log_prob(res.xs)
            - res.post_dist.log_prob(res.zs), 
        dim=SAMPLE_DIM) 
        - torch.log(K)
    ).mean(dim=BATCH_DIM - 1)
