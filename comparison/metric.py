import torch
from comparison.model import VAE
from comparison.loss import IWAE

def IWAE_64(model: VAE, xs: torch.Tensor) -> torch.Tensor:
    with torch.zero_grad():
        vae_res = model(xs, K=64)
        loss = IWAE(vae_res)
        return loss

def log_px(model: VAE, xs: torch.Tensor) -> torch.Tensor:
    with torch.zero_grad():
        vae_res = model(xs, K=5000)
        loss = IWAE(vae_res)
        return loss