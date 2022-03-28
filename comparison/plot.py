import torch
from math import pi, sqrt
import matplotlib.pyplot as plt


def norm_pdf(xs: torch.Tensor, mu: torch.Tensor, sigma: float) -> torch.Tensor:
    """Closed form version of the p.d.f. of a normal distribution."""
    min_mean = xs - mu
    return torch.where(
        sigma == 0,
        (min_mean == 0).float(),
        torch.exp(-min_mean*min_mean/(2*sigma*sigma))/torch.sqrt(2*pi*sigma*sigma),
    )


def gauss_filter(ls: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Smoothen a curve using a Gauss filter.
    
    - `(xs, ys)` represent the original `x` and `y` values
    of the curve.
    - `ls` represents the `x` values of the new smoothened curve.
    - `sigma` represents the std of the Gaussian kernel applied in smoothening.
    Larger `sigma` results in more smoothening.
    """
    weights = norm_pdf(
        ls.view(len(ls), 1),
        xs.view(1, len(xs)),
        sigma
    )
    return torch.matmul(weights, ys) / weights.sum(axis=1)


def gauss_filter_and_std(ls: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor, sigma: float, fit_sigma=False) -> torch.Tensor:
    """
    Returns both the weighted mean and standard deviation as 
    defined by a Gaussian kernel.

    - `(xs, ys)` represent the original `x` and `y` values
    of the curve.
    - `ls` represents the `x` values of the new smoothened curve.
    - `sigma` represents the std of the Gaussian kernel applied in smoothening.
    Larger `sigma` results in more smoothening.
    - `fit_sigma`, when enabled, decreases the std of the kernel near the bounds of `xs` for 
    more visually appealing smoothening.
    """
    def _fit_sigma(xs: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        dist_to_edge = torch.minimum(xs - xs.min(), xs.max() - xs).clamp(min=0.) / 2.
        return torch.minimum(dist_to_edge, sigma)

    sigma = sigma if isinstance(sigma, torch.Tensor) else torch.tensor(sigma)
    weights_ls = norm_pdf(
        ls.view(-1, 1),
        xs.view(1, -1),
        _fit_sigma(ls, sigma).view(-1, 1) if fit_sigma else sigma
    )
    weights_xs = norm_pdf(
        xs.view(-1, 1),
        xs.view(1, -1),
        _fit_sigma(xs, sigma).view(-1, 1) if fit_sigma else sigma
    )
    ys_ls = torch.matmul(weights_ls, ys) / weights_ls.sum(axis=1)
    ys_xs = torch.matmul(weights_xs, ys) / weights_xs.sum(axis=1)
    eps = torch.sqrt(torch.matmul(weights_ls, (ys - ys_xs)**2) / weights_ls.sum(dim=1))
    return ys_ls, eps


def gauss_density(xs: torch.Tensor, ps: torch.Tensor, sigma: float):
    """
    Get the Gaussian Kernel Density Estimate of a series of points.

    - `xs` represent the `x` values we want to determine the KDE of.
    - `ps` represents the list of points we want to plot the density of.
    - `sigma` represents the std of the Gaussian kernel applied in density
    estimation. Larger `sigma` results in less precision, but more overlap. 
    """
    weights = norm_pdf(
        xs.view(-1, 1), 
        ps.view(1, -1),
        torch.tensor(sigma)
    )
    return weights.sum(axis=1) / len(ps)

def plot_smoothed(
    xs: torch.Tensor, 
    ys: torch.Tensor = None, 
    sigma: float = None, 
    fit_sigma = False,
    **kargs
) -> torch.Tensor:
    """
    Plot data with a Gaussian filter applied. 
    """
    
    if sigma == None:
        sigma = (max(xs) - min(xs)) / 10
    if ys == None:
        ys = xs
        xs = torch.arange(len(xs))
        
    ls = torch.linspace(min(xs), max(xs), 100)
    ys_, eps_ = gauss_filter_and_std(ls, xs, ys, sigma, fit_sigma)
    lows = ys_ - eps_
    highs = ys_ + eps_
    
    alpha = kargs["alpha"] if "alpha" in kargs else 1.
    
    plt.plot(ls, ys_, **kargs)
    plt.fill_between(ls, lows, highs, alpha=alpha*0.3, **kargs)

