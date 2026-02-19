"""Robustness score for roundabout policy evaluation.

Uses safety (min distance), stability (jerk, lane changes), efficiency
(speed ratio), on-road fraction, and brake severity. Supports weight
optimization for histogram shaping.
"""

import numpy as np

# Trajectory metrics
SPEED_SCALE = 16.0  # m/s, for avg_speed_ratio

# Normalization scales for compute_robustness
MIN_DIST_SCALE = 20.0  # m, for norm_safety
JERK_SCALE = 0.1  # for norm_stability (jerk sub-component)
LANE_CHANGE_SCALE = 5.0  # typical lane changes per episode
BRAKE_SEVERITY_SCALE = 2.0  # mean (max(0, -a))^2 per step (m^2/s^4)

WEIGHT_KEYS = ("safety", "stability", "efficiency", "road", "hard_brakes")


def _scalar(x):
    """Return float; if x is length-1 array, unwrap it."""
    return (
        float(x[0])
        if hasattr(x, "__len__") and len(x) == 1
        else float(x)
    )


def trajectory_metrics_from_rollout(
    env,
    get_action,
    *,
    seed=None,
    speed_scale=SPEED_SCALE,
):
    """Run one trajectory and return metrics for compute_robustness.

    Resets env (with optional seed), steps until done/truncated, and
    aggregates min_distance, speeds, jerk, brake severity, lane changes,
    on_road fraction, and cumulative reward.

    Args:
        env: Gymnasium-style env with .reset(seed=), .step(action),
            .unwrapped.vehicle, .unwrapped.road, .unwrapped.config.
        get_action: Callable (env, obs) -> action. Used each step.
        seed: Optional int passed to env.reset(seed=).
        speed_scale: Reference speed in m/s for avg_speed_ratio.

    Returns:
        Dict with scalars: success, min_distance, avg_speed_ratio,
        jerk_score, brake_severity, lane_changes, on_road_frac,
        cumulative_reward.
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

    decel = np.clip(-accel, 0, None)
    n = len(decel)
    brake_severity = (
        float(np.sum(decel ** 2) / n) if n > 0 else 0.0
    )

    mean_vel = np.mean(velocities) if len(velocities) > 0 else 0.0
    jerk_score = float(np.mean(jerk ** 2)) if len(jerk) > 0 else 0.0
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


def compute_robustness(
    trajectory_metrics,
    weights=None,
    min_dist_scale=MIN_DIST_SCALE,
    jerk_scale=JERK_SCALE,
    lane_change_scale=LANE_CHANGE_SCALE,
    brake_severity_scale=BRAKE_SEVERITY_SCALE,
):
    """Compute a single-trajectory robustness score in [0, 1].

    Failure (e.g. collision) yields 0. Otherwise a weighted combination
    of normalized safety, stability, efficiency, road, and brake terms.

    Args:
        trajectory_metrics: Dict with min_distance, avg_speed_ratio,
            jerk_score, on_road_frac; optional success, lane_changes,
            brake_severity (default 0).
        weights: Dict with keys safety, stability, efficiency, road,
            hard_brakes. If None, uses 0.2 each.
        min_dist_scale: Scale for min_distance (safety).
        jerk_scale: Scale for jerk (stability).
        lane_change_scale: Scale for lane_changes (stability).
        brake_severity_scale: Scale for brake_severity (hard_brakes).

    Returns:
        Float in (0, 1] for success; 0 on collision/failure.
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

    norm_safety = np.clip(min_dist / min_dist_scale, 0, 1)
    norm_jerk = 1.0 - np.clip(jerk / jerk_scale, 0, 1)
    norm_lane = 1.0 - np.clip(
        lane_changes / lane_change_scale, 0, 1
    )
    norm_stability = (norm_jerk + norm_lane) / 2.0
    norm_efficiency = np.clip(speed, 0, 1)
    norm_road = np.clip(on_road, 0, 1)
    norm_brake = max(
        0.0, 1.0 - brake_severity / brake_severity_scale
    )

    robustness = (
        weights["safety"] * norm_safety
        + weights["stability"] * norm_stability
        + weights["efficiency"] * norm_efficiency
        + weights["road"] * norm_road
        + weights["hard_brakes"] * norm_brake
    )
    return float(robustness)


def weights_from_vector(x):
    """Map a non-negative 5-vector (sum=1) to a weights dict.

    Used by optimizers. Clips to 1e-8 then normalizes.

    Args:
        x: Array-like of length 5 (or same as WEIGHT_KEYS).

    Returns:
        Dict mapping WEIGHT_KEYS to float weights that sum to 1.
    """
    x = np.asarray(x, dtype=float)
    x = np.clip(x, 1e-8, None)
    x = x / x.sum()
    return dict(zip(WEIGHT_KEYS, x))
