from re import L
import torch
import math

from comparison.model import VAEForwardResult
from comparison.examples.vae_mnist import VAE_MNIST


M_SAMPLE_DIM = 0 # Resampling for the same `x` to reduce variance
K_SAMPLE_DIM = 1 # Resampling for the same `x` tighten bound for IWAE

BATCH_DIM = 0 # Also known as "N", sampling over multiple data points `xs`


def var_log_evidence(res: VAEForwardResult, dreg=False) -> torch.Tensor:
    def log_prob_sum(dist, vals) -> torch.Tensor:
        log_probs = dist.log_prob(vals)
        r = log_probs.sum(-1) 
        return r
    
    lqz_x = log_prob_sum(res.post_dist, res.zs)
    if dreg:
        lqz_x = lqz_x.detach()

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


def DREG(res: VAEForwardResult) -> torch.Tensor:
    """
    The IWAE variant of ELBO, with a reformulation of the
    gradient estimator to reduce its variance, and therefore
    SNR.
    """

    log_w = var_log_evidence(res, dreg=True)

    with torch.no_grad():
        importance_weights = (
            log_w - 
            log_w.logsumexp(
                dim=K_SAMPLE_DIM
            ).unsqueeze(K_SAMPLE_DIM)
        ).exp() 
        imp_weights_shaped = importance_weights.clone()
        for _ in range(len(res.zs.shape) - len(imp_weights_shaped.shape)):
            imp_weights_shaped = imp_weights_shaped.unsqueeze(-1)

    if res.zs.requires_grad:
        res.zs.register_hook(lambda grad: imp_weights_shaped * grad)

    return (importance_weights * log_w).sum(
        K_SAMPLE_DIM
    ).mean(
        M_SAMPLE_DIM
    )

def DREG2(res: VAEForwardResult) -> torch.Tensor:
    model = res.model

    def log_prob(dist, vals):
        log_probs = dist.log_prob(vals)
        r = log_probs.sum(-1) 
        return r  

    # Inference

    lp_full = log_prob(res.post_dist, res.zs)
    lp_partial = log_prob(res.post_dist, res.zs.detach())
    log_hat_q_z_x = lp_full + (lp_partial.detach() - lp_partial)


    log_hat_w = sum([
        log_prob(res.prior_dist, res.zs),
    + log_prob(res.lik_dist, res.xs),
    - log_hat_q_z_x
    ])

    # these are a function of nothing
    importance_weights = (log_hat_w - torch.logsumexp(log_hat_w,dim=0)).detach().exp()


    # Generation

    tilde_zs = res.zs.detach()

    log_tilde_p_zs = log_prob(model.prior_dist(), tilde_zs)
    log_tilde_p_x_zs = log_prob(model.lik_dist(tilde_zs), res.xs)

    log_tilde_p_x_and_zs = log_tilde_p_zs + log_tilde_p_x_zs

    # putting it all together    
    
    loss = torch.sum(importance_weights * log_tilde_p_x_and_zs + importance_weights.pow(2) * log_hat_w, dim = K_SAMPLE_DIM).mean(dim=M_SAMPLE_DIM)
    
    return torch.squeeze(loss)



def ELBO_loss(res: VAEForwardResult) -> torch.Tensor:
    return ELBO(res).mean(dim=BATCH_DIM)


def IWAE_loss(res: VAEForwardResult) -> torch.Tensor:
    return IWAE(res).mean(dim=BATCH_DIM)


def CIWAE_loss(res: VAEForwardResult, beta: float) -> torch.Tensor:
    return CIWAE(res, beta).mean(dim=BATCH_DIM)


def PIWAE_loss(res: VAEForwardResult) -> tuple[torch.Tensor, torch.Tensor]:
    miwae, iwae = PIWAE(res)
    return miwae.mean(dim=BATCH_DIM), iwae.mean(dim=BATCH_DIM)

def DREG_loss(res: VAEForwardResult) -> torch.Tensor:
    return DREG2(res).mean(dim=BATCH_DIM)

