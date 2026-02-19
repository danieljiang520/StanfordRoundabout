"""Weight optimization for robustness score.

Maximize mean robustness or fit weights so the nominal robustness
histogram matches an exponential shape. Used by plotting/analysis scripts.
"""

import numpy as np

from src.robustness import (
    BRAKE_SEVERITY_SCALE,
    JERK_SCALE,
    MIN_DIST_SCALE,
    compute_robustness,
    weights_from_vector,
)
from src.robustness_helpers import (
    exponential_cdf,
    random_simplex_point,
    weight_bounds,
)


def _scores_for_weights(
    nominal_metrics,
    weights_dict,
    min_dist_scale,
    jerk_scale,
    brake_severity_scale,
):
    """Return array of compute_robustness scores for given weights."""
    return np.array(
        [
            compute_robustness(
                m,
                weights=weights_dict,
                min_dist_scale=min_dist_scale,
                jerk_scale=jerk_scale,
                brake_severity_scale=brake_severity_scale,
            )
            for m in nominal_metrics
        ]
    )


def optimize_weights(
    nominal_metrics,
    min_dist_scale=MIN_DIST_SCALE,
    jerk_scale=JERK_SCALE,
    brake_severity_scale=BRAKE_SEVERITY_SCALE,
    min_weight=0.05,
):
    """Maximize mean robustness over nominal trajectories.

    Finds weights (sum=1, each >= min_weight) that maximize the mean
    of compute_robustness over the given nominal_metrics.

    Args:
        nominal_metrics: List of trajectory_metrics dicts (success=1).
        min_dist_scale: Passed to compute_robustness.
        jerk_scale: Passed to compute_robustness.
        brake_severity_scale: Passed to compute_robustness.
        min_weight: Minimum per-component weight (default 0.05). Use 0
            to allow near-zero weights.

    Returns:
        Tuple (weights_dict, mean_robustness).

    Raises:
        RuntimeError: If scipy.optimize.minimize does not succeed.
    """
    from scipy.optimize import minimize

    def neg_mean_robustness(x):
        w = weights_from_vector(x)
        scores = _scores_for_weights(
            nominal_metrics,
            w,
            min_dist_scale,
            jerk_scale,
            brake_severity_scale,
        )
        return -float(np.mean(scores))

    x0 = np.ones(5) / 5
    bounds = weight_bounds(min_weight)
    constraints = {"type": "eq", "fun": lambda x: x.sum() - 1.0}
    result = minimize(
        neg_mean_robustness,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=dict(ftol=1e-9, maxiter=500),
    )
    if not result.success:
        raise RuntimeError(
            f"Weight optimization failed: {result.message}"
        )
    weights = weights_from_vector(result.x)
    mean_rob = -result.fun
    return weights, mean_rob


def optimize_weights_for_exponential(
    nominal_metrics,
    lam=2.5,
    num_bins=10,
    min_dist_scale=MIN_DIST_SCALE,
    jerk_scale=JERK_SCALE,
    brake_severity_scale=BRAKE_SEVERITY_SCALE,
    n_restarts=12,
    seed=None,
    min_weight=0.05,
):
    """Fit weights so nominal robustness histogram is exponential-shaped.

    Minimizes MSE between binned robustness and a truncated exponential
    CDF. Uses multiple random restarts; with seed for reproducibility.

    Args:
        nominal_metrics: List of trajectory_metrics dicts (success=1).
        lam: Exponential rate (higher => more mass at high robustness).
        num_bins: Number of histogram bins on [0, 1].
        min_dist_scale: Passed to compute_robustness.
        jerk_scale: Passed to compute_robustness.
        brake_severity_scale: Passed to compute_robustness.
        n_restarts: Number of random initial weight vectors (default 12).
        seed: Optional RNG seed for restarts.
        min_weight: Minimum per-component weight (default 0.05).

    Returns:
        Tuple (weights_dict, loss, scores_array).
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    edges = np.linspace(0, 1, num_bins + 1)
    target_props = np.diff(exponential_cdf(edges, lam))

    def histogram_mse(x):
        w = weights_from_vector(x)
        scores = _scores_for_weights(
            nominal_metrics,
            w,
            min_dist_scale,
            jerk_scale,
            brake_severity_scale,
        )
        counts, _ = np.histogram(scores, bins=edges)
        emp_props = counts / max(scores.size, 1)
        mse = np.sum((emp_props - target_props) ** 2)
        rng_scores = float(np.ptp(scores))
        if rng_scores < 0.15:
            mse += 2.0 * (0.15 - rng_scores)
        return mse

    bounds = weight_bounds(min_weight)
    constraints = {"type": "eq", "fun": lambda x: x.sum() - 1.0}
    opts = dict(ftol=1e-8, maxiter=800)

    starts = [np.ones(5) / 5] + [
        random_simplex_point(5, rng, min_weight)
        for _ in range(n_restarts - 1)
    ]
    best_loss = np.inf
    best_x = None

    for x0 in starts:
        result = minimize(
            histogram_mse,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options=opts,
        )
        if result.fun < best_loss:
            best_loss = result.fun
            best_x = result.x
    if best_x is None:
        best_x = np.ones(5) / 5

    weights = weights_from_vector(best_x)
    scores = _scores_for_weights(
        nominal_metrics,
        weights,
        min_dist_scale,
        jerk_scale,
        brake_severity_scale,
    )
    return weights, float(best_loss), scores
