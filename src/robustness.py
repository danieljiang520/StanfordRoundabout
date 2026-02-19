"""
Robustness score for roundabout policy evaluation.
Uses safety (min distance), stability (jerk, lane changes, brake severity),
efficiency (speed ratio), and on-road fraction.
"""
import numpy as np

# Constants for trajectory metrics (match notebook)
SPEED_SCALE = 16.0  # m/s, for avg_speed_ratio


def trajectory_metrics_from_rollout(
    env,
    get_action,
    *,
    seed=None,
    speed_scale=SPEED_SCALE,
):
    """
    Run one trajectory (reset + step until done/truncated) and return trajectory_metrics.

    env: gymnasium-style env with .reset(seed=), .step(action), .unwrapped.vehicle, .unwrapped.road, .unwrapped.config.
    get_action: callable (env, obs) -> action. Called each step to get the next action.
    seed: optional reset seed.
    speed_scale: reference speed (m/s) for avg_speed_ratio.

    Returns: dict with scalar trajectory metrics expected by compute_robustness:
        min_distance, avg_speed_ratio, jerk_score, on_road_frac,
        plus success, brake_severity, lane_changes, cumulative_reward.
    """
    obs, info = env.reset(seed=seed)
    done = truncated = False
    velocities = []
    min_dist = float("inf")
    lane_changes = 0
    on_road_steps = 0
    total_steps = 0
    cumulative_reward = 0.0
    prev_action = None

    unwrapped = env.unwrapped
    ego = unwrapped.vehicle
    policy_freq = unwrapped.config.get("policy_frequency", 1)

    while not (done or truncated):
        action = get_action(env, obs)
        obs, reward, done, truncated, info = env.step(action)

        cumulative_reward += float(reward)
        total_steps += 1
        ego = unwrapped.vehicle
        v = np.sqrt(ego.velocity[0] ** 2 + ego.velocity[1] ** 2)
        velocities.append(v)

        for vehicle in unwrapped.road.vehicles:
            if vehicle is not ego:
                d = np.linalg.norm(ego.position - vehicle.position)
                min_dist = min(min_dist, d)

        if ego.on_road:
            on_road_steps += 1

        a = int(np.asarray(action).flat[0])
        if prev_action is not None and a != prev_action and a in [0, 2]:
            lane_changes += 1
        prev_action = a

    velocities = np.array(velocities)
    dt = 1.0 / policy_freq
    accel = np.diff(velocities) / dt
    jerk = np.diff(accel) / dt

    # Brake severity: mean squared deceleration (per-step, episode-length invariant)
    decel = np.clip(-accel, 0, None)  # only negative accel (braking)
    n = len(decel)
    brake_severity = float(np.sum(decel**2) / n) if n > 0 else 0.0

    mean_vel = np.mean(velocities) if len(velocities) > 0 else 0.0
    jerk_score = float(np.mean(jerk**2)) if len(jerk) > 0 else 0.0
    on_road_frac = on_road_steps / max(total_steps, 1)

    return {
        "success": 0.0 if unwrapped.vehicle.crashed else 1.0,
        "min_distance": min_dist if min_dist != float("inf") else 0.0,
        "avg_speed_ratio": mean_vel / speed_scale,
        "jerk_score": jerk_score,
        "brake_severity": brake_severity,
        "lane_changes": lane_changes,
        "on_road_frac": on_road_frac,
        "cumulative_reward": cumulative_reward,
    }


# Default normalization constants (from notebook)
MIN_DIST_SCALE = 20.0   # m, for norm_safety
JERK_SCALE = 0.1        # for norm_stability (jerk sub-component)
LANE_CHANGE_SCALE = 5.0  # typical lane changes per episode
BRAKE_SEVERITY_SCALE = 2.0  # mean of max(0, -a)² per step (m²/s⁴)


def _scalar(x):
    return float(x[0]) if hasattr(x, "__len__") and len(x) == 1 else float(x)


def compute_robustness(
    trajectory_metrics,
    weights=None,
    min_dist_scale=MIN_DIST_SCALE,
    jerk_scale=JERK_SCALE,
    lane_change_scale=LANE_CHANGE_SCALE,
    brake_severity_scale=BRAKE_SEVERITY_SCALE,
):
    """
    Compute robustness score for a single trajectory.
    Satisfies f(τ) ≤ 0 for failure trajectories (e.g. collision).

    trajectory_metrics: dict with keys (values are scalars or length-1 sequences):
        "min_distance", "avg_speed_ratio", "jerk_score", "on_road_frac";
        optional "success", "lane_changes", "brake_severity".
    weights: dict with keys "safety", "stability", "efficiency", "road".
        If None, uses uniform weights (0.25 each).
    min_dist_scale: scale for normalizing min_distance (safety).
    jerk_scale, lane_change_scale, brake_severity_scale:
        scales for stability sub-components (lower raw = more stable).

    Returns: float; ≤ 0 on collision, in (0, 1] otherwise.
    """
    success = trajectory_metrics.get("success", 1.0)
    if _scalar(success) <= 0:
        return 0.0

    if weights is None:
        weights = {
            "safety": 0.2,
            "stability": 0.2,
            "efficiency": 0.2,
            "road": 0.2,
            "hard_brakes": 0.2,
        }

    min_dist = _scalar(trajectory_metrics["min_distance"])
    speed = _scalar(trajectory_metrics["avg_speed_ratio"])
    jerk = _scalar(trajectory_metrics["jerk_score"])
    lane_changes = _scalar(trajectory_metrics.get("lane_changes", 0))
    brake_severity = _scalar(trajectory_metrics.get("brake_severity", 0))
    on_road = _scalar(trajectory_metrics["on_road_frac"])
    brake_severity = _scalar(trajectory_metrics["brake_severity"])

    norm_safety = np.clip(min_dist / min_dist_scale, 0, 1)
    # Stability: jerk, lane changes, brake severity (count + magnitude combined)
    norm_jerk = 1.0 - np.clip(jerk / jerk_scale, 0, 1)
    norm_lane = 1.0 - np.clip(lane_changes / lane_change_scale, 0, 1)
    norm_severity = 1.0 - np.clip(brake_severity / brake_severity_scale, 0, 1)
    norm_stability = (norm_jerk + norm_lane + norm_severity) / 3.0
    norm_efficiency = np.clip(speed, 0, 1)
    norm_road = np.clip(on_road, 0, 1)
    norm_brake = max(0.0, 1.0 - brake_severity / brake_severity_scale)

    robustness = (
        weights["safety"] * norm_safety
        + weights["stability"] * norm_stability
        + weights["efficiency"] * norm_efficiency
        + weights["road"] * norm_road
        + weights["hard_brakes"] * norm_brake
    )
    return float(robustness)


WEIGHT_KEYS = ("safety", "stability", "efficiency", "road", "hard_brakes")


def weights_from_vector(x):
    """Map a 5-vector (non-negative, sum=1) to weights dict. For optimization."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-8, None)
    x = x / x.sum()
    return dict(zip(WEIGHT_KEYS, x))


def _scores_for_weights(
    nominal_metrics,
    weights_dict,
    min_dist_scale,
    jerk_scale,
    brake_severity_scale,
):
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


def _weight_bounds(min_weight, n=5):
    """Bounds for n weights that sum to 1, each in [min_weight, 1-(n-1)*min_weight]."""
    if min_weight <= 0:
        return [(1e-6, 1.0)] * n
    max_w = 1.0 - (n - 1) * min_weight
    if max_w < min_weight:
        raise ValueError(f"min_weight too large: need min_weight <= 1/n = {1/n}")
    return [(min_weight, max_w)] * n


def optimize_weights(
    nominal_metrics,
    min_dist_scale=MIN_DIST_SCALE,
    jerk_scale=JERK_SCALE,
    brake_severity_scale=BRAKE_SEVERITY_SCALE,
    min_weight=0.05,
):
    """
    Find weights that maximize mean robustness over nominal (no-collision) trajectories.
    Weights sum to 1; if min_weight > 0, each weight >= min_weight (all components non-zero).

    nominal_metrics: list of trajectory_metrics dicts (success=1 only).
    min_weight: minimum weight per component (default 0.05); use 0 for no minimum.
    Returns: weights dict and final mean robustness.
    """
    from scipy.optimize import minimize

    def neg_mean_robustness(x):
        w = weights_from_vector(x)
        scores = _scores_for_weights(
            nominal_metrics, w, min_dist_scale, jerk_scale, brake_severity_scale
        )
        return -float(np.mean(scores))

    x0 = np.ones(5) / 5
    bounds = _weight_bounds(min_weight)
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
        raise RuntimeError(f"Weight optimization failed: {result.message}")
    weights = weights_from_vector(result.x)
    mean_rob = -result.fun
    return weights, mean_rob


def _exponential_cdf(x, lam):
    """CDF of exponential on [0,1]: F(x) = (exp(lam*x)-1)/(exp(lam)-1)."""
    return (np.exp(lam * x) - 1.0) / (np.exp(lam) - 1.0)


def _random_simplex_point(n, rng=None, min_weight=0.0):
    """Random point on simplex (sum=1, non-negative). If min_weight > 0, each component >= min_weight."""
    rng = rng or np.random.default_rng()
    x = rng.exponential(scale=1.0, size=n)
    x = x / x.sum()
    if min_weight > 0:
        max_w = 1.0 - (n - 1) * min_weight
        x = np.clip(x, min_weight, max_w)
        x = x / x.sum()
    return x


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
    """
    Find weights so the distribution of robustness over nominal trajectories
    matches an exponential shape: low frequency at low robustness, high at high robustness.
    Differentiates "less safe" vs "safer" nominals using the metrics.
    Uses multiple random restarts to avoid getting stuck at uniform weights.
    If min_weight > 0, every weight is >= min_weight (all components non-zero).

    nominal_metrics: list of trajectory_metrics dicts (success=1 only).
    lam: exponential rate (higher => more mass at high robustness).
    num_bins: number of histogram bins on [0, 1].
    n_restarts: number of random initial weight vectors to try (default 12).
    seed: optional RNG seed for restarts.
    min_weight: minimum weight per component (default 0.05); use 0 for no minimum.
    Returns: weights dict, final loss (MSE to target), and scores array.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(seed)
    edges = np.linspace(0, 1, num_bins + 1)
    target_props = np.diff(_exponential_cdf(edges, lam))

    def histogram_mse(x):
        w = weights_from_vector(x)
        scores = _scores_for_weights(
            nominal_metrics, w, min_dist_scale, jerk_scale, brake_severity_scale
        )
        counts, _ = np.histogram(scores, bins=edges)
        emp_props = counts / max(scores.size, 1)
        mse = np.sum((emp_props - target_props) ** 2)
        rng_scores = float(np.ptp(scores))
        if rng_scores < 0.15:
            mse += 2.0 * (0.15 - rng_scores)
        return mse

    bounds = _weight_bounds(min_weight)
    constraints = {"type": "eq", "fun": lambda x: x.sum() - 1.0}
    opts = dict(ftol=1e-8, maxiter=800)

    # Try uniform first, then random restarts (feasible w.r.t. min_weight); keep best
    starts = [np.ones(5) / 5] + [_random_simplex_point(5, rng, min_weight) for _ in range(n_restarts - 1)]
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
        nominal_metrics, weights, min_dist_scale, jerk_scale, brake_severity_scale
    )
    return weights, float(best_loss), scores
