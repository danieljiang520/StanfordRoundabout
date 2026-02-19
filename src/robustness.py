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

    # Brake severity: quadratic penalty on all deceleration (no threshold)
    decel = np.clip(-accel, 0, None)  # only negative accel (braking)
    brake_severity = float(np.sum(decel**2))

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
BRAKE_SEVERITY_SCALE = 100.0  # Σ max(0, -a)² per episode (all deceleration)


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
    # f(τ) ≤ 0 for failure trajectories (e.g. collision)
    success = trajectory_metrics.get("success", 1.0)
    if _scalar(success) <= 0:
        return 0.0

    if weights is None:
        weights = {
            "safety": 0.25,
            "stability": 0.25,
            "efficiency": 0.25,
            "road": 0.25,
        }

    min_dist = _scalar(trajectory_metrics["min_distance"])
    speed = _scalar(trajectory_metrics["avg_speed_ratio"])
    jerk = _scalar(trajectory_metrics["jerk_score"])
    lane_changes = _scalar(trajectory_metrics.get("lane_changes", 0))
    brake_severity = _scalar(trajectory_metrics.get("brake_severity", 0))
    on_road = _scalar(trajectory_metrics["on_road_frac"])

    norm_safety = np.clip(min_dist / min_dist_scale, 0, 1)
    # Stability: jerk, lane changes, brake severity (count + magnitude combined)
    norm_jerk = 1.0 - np.clip(jerk / jerk_scale, 0, 1)
    norm_lane = 1.0 - np.clip(lane_changes / lane_change_scale, 0, 1)
    norm_severity = 1.0 - np.clip(brake_severity / brake_severity_scale, 0, 1)
    norm_stability = (norm_jerk + norm_lane + norm_severity) / 3.0
    norm_efficiency = np.clip(speed, 0, 1)
    norm_road = np.clip(on_road, 0, 1)

    robustness = (
        weights["safety"] * norm_safety
        + weights["stability"] * norm_stability
        + weights["efficiency"] * norm_efficiency
        + weights["road"] * norm_road
    )
    return float(robustness)
