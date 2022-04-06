import torch
import math

from comparison.model import VAEForwardResult


M_SAMPLE_DIM = 0 # Resampling for the same `x` to reduce variance
K_SAMPLE_DIM = 1 # Resampling for the same `x` tighten bound for IWAE

BATCH_DIM = 0 # Also known as "N", sampling over multiple data points `xs`


def var_log_evidence(res: VAEForwardResult) -> torch.Tensor:
    def log_prob_sum(dist, vals):
        log_probs = dist.log_prob(vals)
        r = log_probs.sum(-1) 
        return r
    
    return (
        log_prob_sum(res.prior_dist, res.zs)
        + log_prob_sum(res.lik_dist, res.xs)
        - log_prob_sum(res.post_dist, res.zs)
    )

def logmeanexp(xs: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.logsumexp(xs, dim=dim) - math.log(xs.size(dim))


def ELBO(res: VAEForwardResult) -> torch.Tensor:
    """The original ELBO definition."""
    return var_log_evidence(res).mean(
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
    )


def CIWAE(res: VAEForwardResult, beta: float) -> torch.Tensor:
    """
    Linearly interpolate between original ELBO and IWAE.
    `beta=1` is ELBO, `beta=0` is IWAE.
    """
    beta_tensor = torch.tensor(beta)
    ev = var_log_evidence(res)
    comb = (
        beta_tensor * ev.mean(dim=K_SAMPLE_DIM)
        + (1 - beta_tensor) * logmeanexp(ev, dim=K_SAMPLE_DIM)
    )
    return comb.mean(dim=M_SAMPLE_DIM)


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
    )

    iwae = logmeanexp(
        ev.view(1, -1, *ev.shape[2:]),
        dim=K_SAMPLE_DIM
    ).mean(
        dim=M_SAMPLE_DIM
    )

    return miwae, iwae


def DREG(res: VAEForwardResult) -> tuple[torch.Tensor, torch.Tensor]:
    """
        TODO:
        Actually test this am sure I've got a dim wrong somewhere
        Implement it in the same way as PIWAE on the training side
    """

    def log_prob(dist, vals):
        log_probs = dist.log_prob(vals)
        r = log_probs.sum(-1) 
        return r  

    ev = var_log_evidence(res)

    """
        stop_grad_log_q_z_x should only be a function of z and not of the distribution
        but there is no dist.detach()
        so this is a hack
        it's gonna be slower but hopefull it works
        an alternative would be to open up the model so we can detach mu and std directly
        but this is a p big rewrite of a lot of this code
    """
    
    lp_full = log_prob(res.post_dist, res.zs)
    lp_partial = log_prob(res.post_dist, res.zs.detach())
    stop_grad_log_q_z_x = lp_full + (lp_partial.detach() - lp_partial)

    stop_grad_log_w = sum([
          log_prob(res.prior_dist, res.zs),
        + log_prob(res.lik_dist, res.xs),
        - stop_grad_log_q_z_x
        ])
    

    # these are a function of nothing
    importance_weights = (stop_grad_log_w - torch.logsumexp(stop_grad_log_w,dim=0)).detach().exp()

    infer_loss = - (importance_weights.pow(2) * stop_grad_log_w).sum(0).mean()

    return ev, infer_loss



def ELBO_loss(res: VAEForwardResult) -> torch.Tensor:
    return ELBO(res).mean(dim=BATCH_DIM)


def IWAE_loss(res: VAEForwardResult) -> torch.Tensor:
    return IWAE(res).mean(dim=BATCH_DIM)


def CIWAE_loss(res: VAEForwardResult, beta: float) -> torch.Tensor:
    return CIWAE(res, beta).mean(dim=BATCH_DIM)


def PIWAE_loss(res: VAEForwardResult) -> tuple[torch.Tensor, torch.Tensor]:
    miwae, iwae = PIWAE(res)
    return miwae.mean(dim=BATCH_DIM), iwae.mean(dim=BATCH_DIM)

