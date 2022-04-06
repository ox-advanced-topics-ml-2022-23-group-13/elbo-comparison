from typing import Iterator
import torch
from comparison.model import VAE
from comparison.loss import IWAE, var_log_evidence
from comparison.loss import IWAE_loss, CIWAE_loss, PIWAE_loss


def IWAE_metric(model: VAE, xs: torch.Tensor, M: int = 1, K: int = 64) -> torch.Tensor:
    vae_res = model(xs, K, M)
    return IWAE_loss(vae_res)

def CIWAE_metric(model, xs, beta: float = 0.5) -> torch.Tensor:
    M, K = 1, 64
    vae_res = model(xs, K, M)
    return CIWAE_loss(vae_res, beta)

def PIWAE_metric(model, xs, M = 1, K = 64) -> tuple[torch.Tensor, torch.Tensor]:
    M, K = 1, 64
    vae_res = model(xs, K, M)
    return PIWAE_loss(vae_res)

def IWAE_64(model: VAE, xs: torch.Tensor) -> torch.Tensor:
    vae_res = model(xs, 64, 1)
    return IWAE(vae_res)

def log_px(model: VAE, xs: torch.Tensor) -> torch.Tensor:
    vae_res = model(xs, 5000, 1)
    return IWAE(vae_res)

def sample_grads(
    model: VAE, 
    xss: Iterator[torch.Tensor], 
    params: tuple[torch.nn.parameter.Parameter, ...], 
    M: int, 
    K: int, 
    loss_fn,
    reshape: bool = False
) -> tuple[torch.Tensor, ...]:
    grads = list(None for p in params)

    for p in params:
        p.grad = None

    for xs in xss:
        # for idx in range(xs.size(0)):
        vae_res = model(xs, M=M, K=K)
        loss = -loss_fn(vae_res)

        loss.backward()

        with torch.no_grad():
            for idx, p in enumerate(params):
                grad = torch.unsqueeze(p.grad, dim=0)
                if grads[idx] == None:
                    grads[idx] = grad
                else:
                    grads[idx] = torch.cat([grads[idx], grad], dim=0)
                p.grad = None

    if reshape:
        return torch.cat(tuple(grad.flatten(1, -1) for grad in grads), dim=1)
    else:
        return tuple(grads)

def norm_over(xs: torch.Tensor, dim: int = None) -> torch.Tensor:
    return torch.sqrt((xs ** 2).sum(dim))

def sample_snr(
    model: VAE, 
    xss: Iterator[torch.Tensor], 
    params:  tuple[torch.nn.parameter.Parameter, ...], 
    M: int, 
    K: int, 
    loss_fn
) -> torch.Tensor:
    grads = sample_grads(model, xss, params, M, K, loss_fn, reshape=True)
    return torch.abs(grads.mean(dim=0) / grads.std(dim=0))
    
def sample_dsnr(
    model: VAE, 
    xss: Iterator[torch.Tensor], 
    params:  tuple[torch.nn.parameter.Parameter, ...], 
    M: int, 
    K: int, 
    loss_fn
) -> torch.Tensor:
    grads = sample_grads(model, xss, params, M, K, loss_fn, reshape=True)

    exp = grads.mean(dim=0)
    norm_exp = exp / norm_over(exp)
    
    grads_pll = (grads * norm_exp).sum(dim=1).unsqueeze(dim=-1) * norm_exp
    grads_perp = grads - grads_pll
    dsnr = norm_over(grads_pll, dim=1) / norm_over(grads_perp, dim=1)
    return dsnr.mean()

    
def sample_ess(
    model: VAE,
    xs: torch.Tensor,
    T: int
) -> torch.Tensor:
    vae_res = model(xs, M=T)
    log_weight = var_log_evidence(vae_res).squeeze(1)
    
    effective_sample_sizes = torch.exp(
                                 torch.logsumexp(log_weight, dim=0)*2 
                               - torch.logsumexp(log_weight*2, dim=0)
                             )
    
    return effective_sample_sizes/T

