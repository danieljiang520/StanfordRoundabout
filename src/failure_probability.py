"""Bayesian failure probability estimation from rollout logs.

Uses a Beta posterior to estimate p_fail with uncertainty quantification.
At 33% failure rates, direct/Bayesian estimation is statistically efficient
and preferable over importance sampling or adaptive methods (which are
designed for rare-event settings with pfail << 5%).
"""

from scipy.stats import beta as beta_dist


def estimate_failure_probability(logs, prior_alpha=1.0, prior_beta=1.0, target_delta=None):
    """Estimate failure probability from rollout logs using Bayesian estimation.

    Models failures as Bernoulli(p_fail). With a Beta(a, b) prior, the
    posterior after observing n failures in m rollouts is Beta(a+n, b+m-n).

    Args:
        logs: List of rollout result dicts, each with key "is_failure" (bool).
        prior_alpha: Alpha parameter of Beta prior (default 1.0 = uniform).
        prior_beta: Beta parameter of Beta prior (default 1.0 = uniform).
        target_delta: Optional float. If provided, computes P(p_fail < target_delta).

    Returns:
        Dict with:
            n_failures: Number of observed failures.
            n_rollouts: Total number of rollouts.
            mle: Maximum likelihood estimate n_failures / n_rollouts.
            posterior_mean: Bayesian posterior mean.
            posterior_std: Bayesian posterior standard deviation.
            ci_95_low: 2.5th percentile of posterior (lower bound of 95% CI).
            ci_95_high: 97.5th percentile of posterior (upper bound of 95% CI).
            bound_95: 95th percentile of posterior (one-sided upper bound).
            p_below_delta: P(p_fail < target_delta) if target_delta provided, else None.
            posterior_alpha: Posterior alpha parameter.
            posterior_beta: Posterior beta parameter.
    """
    n = sum(1 for r in logs if r["is_failure"])
    m = len(logs)

    post_alpha = prior_alpha + n
    post_beta = prior_beta + (m - n)
    posterior = beta_dist(post_alpha, post_beta)

    mle = n / m if m > 0 else 0.0
    post_mean = posterior.mean()
    post_std = posterior.std()
    ci_low = posterior.ppf(0.025)
    ci_high = posterior.ppf(0.975)
    bound_95 = posterior.ppf(0.95)

    p_below = posterior.cdf(target_delta) if target_delta is not None else None

    return {
        "n_failures": n,
        "n_rollouts": m,
        "mle": mle,
        "posterior_mean": post_mean,
        "posterior_std": post_std,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "bound_95": bound_95,
        "p_below_delta": p_below,
        "posterior_alpha": post_alpha,
        "posterior_beta": post_beta,
    }


def print_failure_probability_report(result, target_delta=None):
    """Print a human-readable summary of the failure probability estimate.

    Args:
        result: Dict returned by estimate_failure_probability().
        target_delta: Optional safety threshold to report confidence against.
    """
    n, m = result["n_failures"], result["n_rollouts"]
    print(f"Failure Probability Estimation ({n} failures in {m} rollouts)")
    print(f"  MLE (point estimate):    {result['mle']:.4f}")
    print(f"  Bayesian posterior mean: {result['posterior_mean']:.4f}  ± {result['posterior_std']:.4f} (std)")
    print(f"  95% credible interval:   [{result['ci_95_low']:.4f}, {result['ci_95_high']:.4f}]")
    print(f"  95% one-sided bound:     p_fail < {result['bound_95']:.4f} with 95% confidence")
    if result["p_below_delta"] is not None:
        d = target_delta
        print(f"  P(p_fail < {d}):          {result['p_below_delta']:.4f}")
