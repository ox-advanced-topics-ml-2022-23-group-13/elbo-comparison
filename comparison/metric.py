from typing import Iterator
import torch
from comparison.model import VAE
from comparison.loss import IWAE

def IWAE_64(model: VAE, xs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        vae_res = model(xs, K=64)
        loss = IWAE(vae_res)
        return loss

def log_px(model: VAE, xs: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        vae_res = model(xs, K=5000)
        loss = IWAE(vae_res)
        return loss

def sample_grads(
    model: VAE, 
    xss: Iterator[torch.Tensor], 
    params: tuple[torch.nn.Parameter, ...], 
    M: int, 
    K: int, 
    loss_fn,
    reshape: bool = False
) -> tuple[torch.Tensor, ...]:
    grads = list(None for p in params)

    for xs in xss:
        for idx in range(xs.size(0)):
            x = xs[idx:idx+1]
            vae_res = model(x, M=M, K=K)
            loss = loss_fn(vae_res)

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
    params:  tuple[torch.nn.Parameter, ...], 
    M: int, 
    K: int, 
    loss_fn
) -> torch.Tensor:
    grads = sample_grads(model, xss, params, M, K, loss_fn, reshape=True)

    exp = grads.mean(dim=0)
    std = torch.sqrt(((grads - exp) ** 2).mean(dim=0))
    snr = exp / std
    return norm_over(snr)
    
def sample_dsnr(
    model: VAE, 
    xss: Iterator[torch.Tensor], 
    params:  tuple[torch.nn.Parameter, ...], 
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

    