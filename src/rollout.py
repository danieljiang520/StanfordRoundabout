"""
Rollout function for trajectory simulation with log-probability computation.
"""
from typing import Optional

import numpy as np
import torch

from .distributions import gaussian_mixture_pdf, sample_gaussian_mixture, to_tensor
from .robustness import compute_robustness
from .scenario_params import NOMINAL, ScenarioParams


def rollout(
    env,
    model,
    robustness_weights: Optional[dict] = None,
    eps: float = 1e-10,
    seed: Optional[int] = None,
    speed_scale: float = 16.0,
) -> dict:
    """Run one episode with noise injection and compute log-probability.
    
    Runs the trained policy on the environment while:
    - Adding observation noise (position, velocity) from scenario params
    - Optionally fuzzing actions with random replacements
    - Computing the log-probability of the trajectory under the distributions
    - Computing robustness metrics
    
    Args:
        env: Gymnasium environment (should have unwrapped.scenario_params).
        model: Trained model with .predict(obs, deterministic=True) method.
        robustness_weights: Optional dict of weights for robustness computation.
        eps: Small epsilon for numerical stability in log computations.
        seed: Optional seed for environment reset.
        speed_scale: Reference speed for avg_speed_ratio normalization.
    
    Returns:
        Dictionary containing:
            - log_prob: Log-probability of trajectory under scenario params
            - nominal_log_prob: Log-probability under NOMINAL params
            - is_failure: Whether the ego vehicle crashed
            - robustness: Computed robustness score
    """
    log_prob = 0.0
    nominal_log_prob = 0.0

    env_params = env.unwrapped.scenario_params

    # Get action noise probability
    p_noise = env_params.high_lvl_ctrl_noise.p
    if isinstance(p_noise, torch.nn.Parameter) or torch.is_tensor(p_noise):
        p_noise = torch.clamp(p_noise, 0.0, 1.0).item()
    else:
        p_noise = p_noise[0]

    # Sample observation noise for each car
    num_cars = 4
    noise_position_x = torch.tensor(
        [sample_gaussian_mixture(env_params.initial_position_x) for _ in range(num_cars)]
    )
    noise_position_y = torch.tensor(
        [sample_gaussian_mixture(env_params.initial_position_y) for _ in range(num_cars)]
    )
    noise_velocity_x = torch.tensor(
        [sample_gaussian_mixture(env_params.velocity_x) for _ in range(num_cars)]
    )
    noise_velocity_y = torch.tensor(
        [sample_gaussian_mixture(env_params.velocity_y) for _ in range(num_cars)]
    )

    # Environment parameters log-prob
    for i in range(1, 5):
        v = torch.tensor(env.unwrapped.road.vehicles[i].speed)

        # Speed log-prob
        log_prob += torch.log(
            torch.clamp(
                torch.distributions.Normal(
                    env_params.other_vehicle_speed.mu,
                    env_params.other_vehicle_speed.sigma,
                ).log_prob(v)
                + eps,
                eps,
            )
        )
        nominal_log_prob += torch.log(
            torch.clamp(
                torch.distributions.Normal(
                    NOMINAL.other_vehicle_speed.mu,
                    NOMINAL.other_vehicle_speed.sigma,
                ).log_prob(v)
                + eps,
                eps,
            )
        )

        # Politeness log-prob
        pol_tensor = torch.tensor(env_params.politeness.ab, dtype=torch.float)
        alpha, beta = pol_tensor[0], pol_tensor[0] + pol_tensor[1]
        log_prob += torch.distributions.Beta(alpha, beta).log_prob(
            env.unwrapped.road.vehicles[i].POLITENESS
        )

        nominal_pol = NOMINAL.politeness.ab
        nominal_alpha, nominal_beta = nominal_pol[0], nominal_pol[0] + nominal_pol[1]
        nominal_log_prob += torch.distributions.Beta(nominal_alpha, nominal_beta).log_prob(
            env.unwrapped.road.vehicles[i].POLITENESS
        )

    # Entering vehicle position log-prob
    log_prob += torch.log(
        torch.clamp(
            gaussian_mixture_pdf(
                env.unwrapped.road.vehicles[-1].position[0],
                env_params.entering_vehicle_position,
            )
            + eps,
            eps,
        )
    )
    nominal_log_prob += torch.log(
        torch.clamp(
            gaussian_mixture_pdf(
                env.unwrapped.road.vehicles[-1].position[0],
                NOMINAL.entering_vehicle_position,
            )
            + eps,
            eps,
        )
    )

    # Ego vehicle initial speed log-prob
    log_prob += torch.distributions.Normal(
        env_params.initial_speed.mu, env_params.initial_speed.sigma
    ).log_prob(torch.tensor(env.unwrapped.vehicle.speed))
    nominal_log_prob += torch.distributions.Normal(
        NOMINAL.initial_speed.mu, NOMINAL.initial_speed.sigma
    ).log_prob(torch.tensor(env.unwrapped.vehicle.speed))

    # Ego heading log-prob
    head_probs = torch.tensor(env_params.initial_heading.p, dtype=torch.float32)
    last_prob = 1.0 - head_probs.sum()
    probs = torch.cat([head_probs, last_prob.unsqueeze(0)])
    log_prob += torch.log(probs[env.unwrapped.road.vehicles[0].heading_idx] + eps)

    nominal_head_probs = torch.tensor(NOMINAL.initial_heading.p, dtype=torch.float32)
    nominal_last_prob = 1.0 - nominal_head_probs.sum()
    nominal_probs = torch.cat([nominal_head_probs, nominal_last_prob.unsqueeze(0)])
    nominal_log_prob += torch.log(
        nominal_probs[env.unwrapped.road.vehicles[0].heading_idx] + eps
    )

    # Episode loop
    trajectory = []
    done = truncated = False
    obs, info = env.reset()

    # Robustness metric tracking
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
        # Add sensor noise
        obs_noisy = obs.copy()
        for i in range(num_cars):
            obs_noisy[i + 1, 1] += noise_position_x[i]
            obs_noisy[i + 1, 2] += noise_position_y[i]
            obs_noisy[i + 1, 3] += noise_velocity_x[i]
            obs_noisy[i + 1, 4] += noise_velocity_y[i]

        action, _states = model.predict(obs_noisy, deterministic=True)

        # High-level action fuzzing
        if torch.rand(1).item() < p_noise:
            available = unwrapped.action_type.get_available_actions()
            action = unwrapped.np_random.choice(available)
            action_prob = p_noise / len(available)
            nominal_action_prob = NOMINAL.high_lvl_ctrl_noise.p[0] / len(available)
        else:
            action_prob = 1.0 - p_noise
            nominal_action_prob = 1 - NOMINAL.high_lvl_ctrl_noise.p[0]

        log_prob += torch.log(torch.clamp(torch.tensor(action_prob) + eps, eps))
        nominal_log_prob += torch.log(
            torch.clamp(torch.tensor(nominal_action_prob) + eps, eps)
        )

        # Observation noise log-prob
        for i in range(1, num_cars + 1):
            log_prob += torch.log(
                torch.clamp(
                    gaussian_mixture_pdf(obs_noisy[i, 1], env_params.initial_position_x)
                    + eps,
                    eps,
                )
            )
            log_prob += torch.log(
                torch.clamp(
                    gaussian_mixture_pdf(obs_noisy[i, 2], env_params.initial_position_y)
                    + eps,
                    eps,
                )
            )
            log_prob += torch.log(
                torch.clamp(
                    gaussian_mixture_pdf(obs_noisy[i, 3], env_params.velocity_x) + eps, eps
                )
            )
            log_prob += torch.log(
                torch.clamp(
                    gaussian_mixture_pdf(obs_noisy[i, 4], env_params.velocity_y) + eps, eps
                )
            )

            nominal_log_prob += torch.log(
                torch.clamp(
                    gaussian_mixture_pdf(obs_noisy[i, 1], NOMINAL.initial_position_x)
                    + eps,
                    eps,
                )
            )
            nominal_log_prob += torch.log(
                torch.clamp(
                    gaussian_mixture_pdf(obs_noisy[i, 2], NOMINAL.initial_position_y)
                    + eps,
                    eps,
                )
            )
            nominal_log_prob += torch.log(
                torch.clamp(
                    gaussian_mixture_pdf(obs_noisy[i, 3], NOMINAL.velocity_x) + eps, eps
                )
            )
            nominal_log_prob += torch.log(
                torch.clamp(
                    gaussian_mixture_pdf(obs_noisy[i, 4], NOMINAL.velocity_y) + eps, eps
                )
            )

        trajectory.append(obs_noisy)

        # Step environment
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

        env.render()

    # Compute trajectory metrics
    velocities = np.array(velocities)
    dt = 1.0 / policy_freq
    accel = np.diff(velocities) / dt
    jerk = np.diff(accel) / dt

    decel = np.clip(-accel, 0, None)
    n = len(decel)
    brake_severity = float(np.sum(decel**2) / n) if n > 0 else 0.0

    mean_vel = np.mean(velocities) if len(velocities) > 0 else 0.0
    jerk_score = float(np.mean(jerk**2)) if len(jerk) > 0 else 0.0
    on_road_frac = on_road_steps / max(total_steps, 1)

    trajectory_metrics = {
        "success": 0.0 if unwrapped.vehicle.crashed else 1.0,
        "min_distance": min_dist if min_dist != float("inf") else 0.0,
        "avg_speed_ratio": mean_vel / speed_scale,
        "jerk_score": jerk_score,
        "brake_severity": brake_severity,
        "lane_changes": lane_changes,
        "on_road_frac": on_road_frac,
        "cumulative_reward": cumulative_reward,
    }

    env.close()

    return {
        "log_prob": log_prob.item(),
        "nominal_log_prob": nominal_log_prob.item(),
        "is_failure": trajectory_metrics["success"] < 1e-2,
        "robustness": compute_robustness(trajectory_metrics, weights=robustness_weights),
    }
