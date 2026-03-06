"""
Distribution helper functions for sampling and probability calculations.
"""
import math
from typing import Union,Optional

import torch
import torch.nn as nn

from .scenario_params import GaussianMixtureParam, ParamType


def to_tensor(param: ParamType, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert scalar, list, or nn.Parameter to a torch tensor.
    
    Args:
        param: Input value (float, int, list, Tensor, or nn.Parameter).
        dtype: Target dtype for the tensor.
    
    Returns:
        Torch tensor of the specified dtype.
    """
    if isinstance(param, torch.Tensor):
        return param.to(dtype)
    if isinstance(param, nn.Parameter):
        return param.data.to(dtype)
    if isinstance(param, (list, tuple)):
        return torch.tensor(param, dtype=dtype)
    return torch.tensor([param], dtype=dtype)


def sample_gaussian_mixture(param: GaussianMixtureParam, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Sample a single value from a Gaussian mixture.
    
    Args:
        param: GaussianMixtureParam with p, mu, sigma.
    
    Returns:
        A scalar tensor sampled from the mixture.
    """
    p = to_tensor(param.p)
    mu = to_tensor(param.mu)
    sigma = to_tensor(param.sigma)

    # Pad mixture probabilities if necessary
    if len(p) < len(mu):
        last_prob = 1.0 - p.sum()
        p = torch.cat([p, last_prob.unsqueeze(0)])

    # Choose a mixture component
    mixture_idx = torch.distributions.Categorical(p).sample().item()

    # Sample from selected Gaussian
    sample = mu[mixture_idx] + sigma[mixture_idx] * torch.randn(1, generator = generator)
    return sample.squeeze()


def gaussian_pdf(
    x: Union[float, torch.Tensor],
    mu: Union[float, torch.Tensor],
    sigma: Union[float, torch.Tensor],
) -> torch.Tensor:
    """Compute the PDF of a univariate Gaussian at x.
    
    Args:
        x: Point(s) at which to evaluate the PDF.
        mu: Mean of the Gaussian.
        sigma: Standard deviation of the Gaussian.
    
    Returns:
        PDF value(s) at x.
    """
    return (1.0 / (sigma * math.sqrt(2 * math.pi))) * torch.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def gaussian_mixture_pdf(
    x: Union[float, torch.Tensor],
    param: GaussianMixtureParam,
) -> torch.Tensor:
    """Compute the PDF of a Gaussian mixture at x.
    
    Args:
        x: Point(s) at which to evaluate the PDF.
        param: GaussianMixtureParam with p, mu, sigma.
    
    Returns:
        PDF value(s) at x.
    """
    p = torch.tensor(param.p)
    mu = torch.tensor(param.mu)
    sigma = torch.tensor(param.sigma)

    # Pad probabilities if needed
    if len(p) < len(mu):
        last_prob = 1.0 - p.sum()
        p = torch.cat([p, last_prob.unsqueeze(0)])

    pdf = 0.0
    for w, m, s in zip(p, mu, sigma):
        pdf += w * gaussian_pdf(x, m, s)
    return pdf
