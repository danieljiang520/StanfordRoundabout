"""Failure probability estimation from rollout logs.

Provides three estimators (textbook Ch. 7):
  - Direct / MLE  (Eq 7.2)
  - Bayesian Beta-Bernoulli posterior  (Eq 7.4, Algorithm 7.2)
  - Importance sampling  (Eq 7.12, Algorithm 7.3)
"""

import numpy as np
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


def importance_sampling_estimate(is_logs, self_normalized=False):
    """Estimate failure probability via importance sampling (textbook Eq 7.12).

    Trajectories must be sampled from a proposal distribution q(τ) that
    differs from the nominal p(τ).  Each log entry needs ``log_prob``
    (log q), ``nominal_log_prob`` (log p), and ``is_failure``.

    Standard IS  (Eq 7.12):  p_fail ≈ (1/m) Σ w_i · 1{failure_i}
    Self-norm IS (Eq 7.33):  p_fail ≈ Σ w_i·1{fail_i} / Σ w_i

    where w_i = p(τ_i) / q(τ_i) = exp(nominal_log_prob - log_prob).

    Args:
        is_logs: List of rollout dicts from a proposal distribution.
            Each must contain 'log_prob', 'nominal_log_prob', 'is_failure'.
        self_normalized: If True, use self-normalized IS (Eq 7.33 /
            Algorithm 7.9). More robust when log-prob densities are
            approximate, but introduces a small bias.

    Returns:
        Dict with p_fail_is, is_std, ess, n_proposal_failures, n_rollouts,
        weights.
    """
    m = len(is_logs)
    empty = {"p_fail_is": 0.0, "is_std": 0.0, "ess": 0.0,
             "n_proposal_failures": 0, "n_rollouts": 0, "weights": np.array([])}
    if m == 0:
        return empty

    log_w = np.array([r["nominal_log_prob"] - r["log_prob"] for r in is_logs])
    failures = np.array([1.0 if r["is_failure"] else 0.0 for r in is_logs])

    if self_normalized:
        # Shift for numerical stability — cancels in the ratio (Eq 7.33)
        log_w_stable = log_w - np.max(log_w)
        w = np.exp(log_w_stable)
        w_norm = w / np.sum(w)
        p_fail_is = float(np.sum(w_norm * failures))
    else:
        # Standard IS (Eq 7.12) — needs true weights, no shift allowed
        w = np.exp(np.clip(log_w, -500, 500))
        p_fail_is = float(np.mean(w * failures))

    # Standard error of the IS estimator
    is_var = float(np.var(w * failures, ddof=1) / m)
    is_std = float(np.sqrt(max(is_var, 0.0)))

    # Effective sample size (Exercise 7.7): ESS = (Σ w_i·f_i)² / Σ (w_i·f_i)²
    wf = w * failures
    wf_sum = np.sum(wf)
    ess = float(wf_sum ** 2 / np.sum(wf ** 2)) if wf_sum > 0 else 0.0

    return {
        "p_fail_is": p_fail_is,
        "is_std": is_std,
        "ess": ess,
        "n_proposal_failures": int(np.sum(failures)),
        "n_rollouts": m,
        "weights": w,
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
