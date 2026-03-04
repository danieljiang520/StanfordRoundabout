"""
Fuzzer for finding critical failure scenarios in trajectory simulation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import minimize, OptimizeResult

from .distributions import gaussian_mixture_pdf, sample_gaussian_mixture
from .robustness import compute_robustness
from .scenario_params import NOMINAL, SQRT_2, ScenarioParams


@dataclass
class FuzzerConfig:
    """Configuration for the ScenarioFuzzer.
    
    Attributes:
        n_samples: Number of rollouts to average per evaluation.
        log_prob_weight: Weight for nominal log-prob penalty in objective.
        verbose: Whether to print progress during optimization.
        bounds: Parameter bounds as list of (low, high) tuples.
        initial_guess: Initial parameter values for optimization.
    """
    n_samples: int = 3
    log_prob_weight: float = 0.01
    verbose: bool = True
    bounds: List[Tuple[float, float]] = field(default_factory=lambda: [
        (0, 150),      # entering_vehicle_mu
        (10, 25),      # other_vehicle_speed
        (0.0, 0.5),    # ctrl_noise_p
        (0.0, 16.0),   # initial_speed_mu
        (0.1, 10.0),   # politeness_alpha
        (0.1, 10.0),   # politeness_beta
        (0.001, 0.1),  # pos_x_sigma
        (0.001, 0.1),  # pos_y_sigma
        (0.001, 0.1),  # vel_x_sigma
        (0.001, 0.1),  # vel_y_sigma
    ])
    initial_guess: np.ndarray = field(
        default_factory=lambda: np.array([
            5.0, 16.0, 0.0, 8.0, 1.0, 1.0, 0.005, 0.005, 0.005, 0.005
        ])
    )


PARAM_NAMES = [
    "entering_vehicle_mu",
    "other_vehicle_speed",
    "ctrl_noise_p",
    "initial_speed_mu",
    "politeness_alpha",
    "politeness_beta",
    "pos_x_sigma",
    "pos_y_sigma",
    "vel_x_sigma",
    "vel_y_sigma",
]


class ScenarioFuzzer:
    """Fuzzer for finding critical failure scenarios.
    
    Searches for scenario parameters that minimize trajectory robustness
    while keeping the scenarios plausible (high nominal log-probability).
    
    Example:
        >>> fuzzer = ScenarioFuzzer(env, model, setup)
        >>> result = fuzzer.rollout()  # single rollout
        >>> opt_result = fuzzer.optimize()  # find critical scenarios
    """
    
    def __init__(
        self,
        env,
        model,
        scenario_params: Optional[ScenarioParams] = None,
        robustness_weights: Optional[Dict[str, float]] = None,
        config: Optional[FuzzerConfig] = None,
    ):
        """Initialize the fuzzer.
        
        Args:
            env: Gymnasium environment with scenario_params attribute.
            model: Trained policy model (e.g., DQN).
            scenario_params: ScenarioParams object to modify during fuzzing.
                If None, uses env.unwrapped.scenario_params.
            robustness_weights: Optional weights for robustness computation.
            config: FuzzerConfig with optimization settings. Uses defaults if None.
        """
        self.env = env
        self.model = model
        self.scenario_params = scenario_params or env.unwrapped.scenario_params
        self.robustness_weights = robustness_weights
        self.config = config or FuzzerConfig()
        self._eval_count = 0
        self._history: List[Dict[str, Any]] = []
    
    def rollout(
        self,
        eps: float = 1e-10,
        speed_scale: float = 16.0,
    ) -> dict:
        """Run one episode with noise injection and compute log-probability.
        
        Args:
            eps: Small epsilon for numerical stability in log computations.
            speed_scale: Reference speed for avg_speed_ratio normalization.
        
        Returns:
            Dictionary containing:
                - log_prob: Log-probability of trajectory under scenario params
                - nominal_log_prob: Log-probability under NOMINAL params
                - is_failure: Whether the ego vehicle crashed
                - robustness: Computed robustness score
                - metrics: Dict of individual trajectory metrics
        """
        env = self.env
        model = self.model
        env_params = env.unwrapped.scenario_params
        
        log_prob = 0.0
        nominal_log_prob = 0.0

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
            "robustness": compute_robustness(trajectory_metrics, weights=self.robustness_weights),
            "metrics": trajectory_metrics,
        }
    
    def _apply_params(self, x: np.ndarray) -> None:
        """Apply parameter vector to scenario_params."""
        setup = self.scenario_params
        setup.entering_vehicle_position.mu = [x[0], x[0]]
        setup.other_vehicle_speed.mu = x[1]
        setup.high_lvl_ctrl_noise.p = [np.clip(x[2], 0, 1)]
        setup.initial_speed.mu = x[3]
        setup.politeness.ab = [x[4], x[5]]
        setup.initial_position_x.sigma = [x[6], x[6]]
        setup.initial_position_y.sigma = [x[7], x[7]]
        setup.velocity_x.sigma = [SQRT_2 * x[8] / 0.1, SQRT_2 * x[8] / 0.1]
        setup.velocity_y.sigma = [SQRT_2 * x[9] / 0.1, SQRT_2 * x[9] / 0.1]
        self.env.unwrapped.scenario_params = setup
    
    def objective(self, x: np.ndarray) -> float:
        """Compute objective value for parameter vector.
        
        Minimizes robustness while penalizing low nominal log-probability.
        
        Args:
            x: Parameter vector of length 10.
        
        Returns:
            Objective value (lower means more critical failure scenario).
        """
        self._apply_params(x)
        
        results = [self.rollout() for _ in range(self.config.n_samples)]
        
        avg_robustness = np.mean([r["robustness"] for r in results])
        avg_nominal_log_prob = np.mean([r["nominal_log_prob"] for r in results])
        
        obj = avg_robustness - (self.config.log_prob_weight * avg_nominal_log_prob)
        
        self._eval_count += 1
        record = {
            "eval": self._eval_count,
            "params": x.copy(),
            "robustness": avg_robustness,
            "nominal_log_prob": avg_nominal_log_prob,
            "objective": obj,
        }
        self._history.append(record)
        
        if self.config.verbose:
            print(
                f"[{self._eval_count}] Rob: {avg_robustness:.4f} | "
                f"LogProb: {avg_nominal_log_prob:.2f} | Obj: {obj:.4f}"
            )
        
        return float(obj)
    
    def optimize(
        self,
        method: str = "L-BFGS-B",
        maxiter: Optional[int] = None,
        **minimize_kwargs,
    ) -> OptimizeResult:
        """Run optimization to find critical failure scenarios.
        
        Args:
            method: Optimization method (default: L-BFGS-B).
            maxiter: Maximum iterations. If None, uses scipy default.
            **minimize_kwargs: Additional kwargs passed to scipy.optimize.minimize.
        
        Returns:
            scipy.optimize.OptimizeResult with optimal parameters.
        """
        self._eval_count = 0
        self._history = []
        
        options = minimize_kwargs.pop("options", {})
        if maxiter is not None:
            options["maxiter"] = maxiter
        
        result = minimize(
            self.objective,
            self.config.initial_guess,
            bounds=self.config.bounds,
            method=method,
            options=options if options else None,
            **minimize_kwargs,
        )
        return result
    
    def get_params_dict(self, x: np.ndarray) -> Dict[str, float]:
        """Convert parameter vector to named dictionary."""
        return dict(zip(PARAM_NAMES, x))
    
    @property
    def history(self) -> List[Dict[str, Any]]:
        """Return evaluation history from most recent optimize() call."""
        return self._history
    
    @property
    def best_result(self) -> Optional[Dict[str, Any]]:
        """Return the best (lowest objective) result from history."""
        if not self._history:
            return None
        return min(self._history, key=lambda r: r["objective"])
