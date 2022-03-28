import torch
import math

from comparison.model import VAEForwardResult


M_SAMPLE_DIM = 0 # Resampling for the same `x` to reduce variance
K_SAMPLE_DIM = 1 # Resampling for the same `x` tighten bound for IWAE
BATCH_DIM = 2 # Also known as "N", sampling over multiple data points `xs`


def var_log_evidence(res: VAEForwardResult) -> torch.Tensor:
    def log_prob_sum(dist, vals, samples):
        r = dist.log_prob(vals).view(*vals.shape[:3 if samples else 1], -1).sum(-1)
        return r
    return (
        log_prob_sum(res.prior_dist, res.zs, samples=True)
        + log_prob_sum(res.lik_dist, res.xs, samples=False)
        - log_prob_sum(res.post_dist, res.zs, samples=True)
    )

def logmeanexp(xs: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(xs, dim=dim) - math.log(xs.size(dim))


def ELBO(res: VAEForwardResult) -> torch.Tensor:
    """The original ELBO definition."""
    return var_log_evidence(res).mean(
        dim=BATCH_DIM
    ).mean(
        dim=K_SAMPLE_DIM
    ).mean(
        dim=M_SAMPLE_DIM
    )


def IWAE(res: VAEForwardResult) -> torch.Tensor:
    """
    The IWAE variant of ELBO, which tightens bound with 
    log-evidence arbitraily as K increases.

    MIWAE is also captured by this function for values M, K > 1.
    """
    return logmeanexp(
        var_log_evidence(res),
        dim=K_SAMPLE_DIM
    ).mean(
        dim=M_SAMPLE_DIM
    ).mean(
        dim=BATCH_DIM - 2
    )


def CIWAE(res: VAEForwardResult, beta: float) -> torch.Tensor:
    """
    Linearly interpolate between original ELBO and IWAE.
    `beta=1` is ELBO, `beta=0` is IWAE.
    """
    beta = torch.tensor(beta)
    ev = var_log_evidence(res)
    comb = (
        beta * ev.mean(dim=K_SAMPLE_DIM)
        + (1 - beta) * logmeanexp(ev, dim=K_SAMPLE_DIM)
    )
    return comb.mean(dim=M_SAMPLE_DIM).mean(dim=BATCH_DIM - 2)


def PIWAE(res: VAEForwardResult) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Optimise the encoder (inference) network using MIWAE, and 
    optimise the decoder (generative) network using IWAE.

    This function returns the valuations of both, (in the case of IWAE
    treating samples in the M dimension as if they were K), and it is 
    the role of the optimizer to apply gradients accordingly.
    """
    ev = var_log_evidence(res)

    miwae = logmeanexp(
        ev,
        dim=K_SAMPLE_DIM
    ).mean(
        dim=M_SAMPLE_DIM
    ).mean(
        dim=BATCH_DIM - 2
    )

    iwae = logmeanexp(
        ev.view(1, -1, *ev.shape[2:]),
        dim=K_SAMPLE_DIM
    ).mean(
        dim=M_SAMPLE_DIM
    ).mean(
        dim=BATCH_DIM - 2
    )

    return miwae, iwae

