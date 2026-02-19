"""Internal helpers for robustness weight optimization.

This module holds pure-math utilities used by src.robustness_optimize
(weight_bounds, exponential_cdf, random_simplex_point).
"""

import numpy as np


def weight_bounds(min_weight, n=5):
    """Bounds for n weights that sum to 1.

    Each weight is in [min_weight, 1 - (n-1)*min_weight] when min_weight > 0.

    Args:
        min_weight: Minimum value per weight. If <= 0, returns (1e-6, 1.0) per dim.
        n: Number of weights (default 5).

    Returns:
        List of (low, high) tuples for each of the n weights.

    Raises:
        ValueError: If min_weight is too large (e.g. > 1/n).
    """
    if min_weight <= 0:
        return [(1e-6, 1.0)] * n
    max_w = 1.0 - (n - 1) * min_weight
    if max_w < min_weight:
        raise ValueError(
            f"min_weight too large: need min_weight <= 1/n = {1/n}"
        )
    return [(min_weight, max_w)] * n


def exponential_cdf(x, lam):
    """CDF of truncated exponential on [0, 1].

    F(x) = (exp(lam * x) - 1) / (exp(lam) - 1).

    Args:
        x: Array or scalar in [0, 1].
        lam: Rate parameter.

    Returns:
        CDF values, same shape as x.
    """
    return (np.exp(lam * x) - 1.0) / (np.exp(lam) - 1.0)


def random_simplex_point(n, rng=None, min_weight=0.0):
    """Sample a random point on the simplex (sum=1, non-negative).

    If min_weight > 0, each component is at least min_weight (and
    renormalized to sum to 1).

    Args:
        n: Dimension (number of components).
        rng: Optional np.random.Generator. If None, uses default_rng().
        min_weight: Optional minimum per component (default 0).

    Returns:
        One-dimensional array of length n that sums to 1.
    """
    rng = rng or np.random.default_rng()
    x = rng.exponential(scale=1.0, size=n)
    x = x / x.sum()
    if min_weight > 0:
        max_w = 1.0 - (n - 1) * min_weight
        x = np.clip(x, min_weight, max_w)
        x = x / x.sum()
    return x
