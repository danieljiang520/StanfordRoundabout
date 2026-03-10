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
        seed: int = 42,
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
        
        # Set random seeds for reproducibility
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            # Torch generator used for sampling in this class
            self.torch_rng = torch.Generator().manual_seed(seed)

            # Seed environment's RNG in a way that works for both
            # legacy RandomState and new NumPy Generator APIs
            if hasattr(env.unwrapped, "np_random"):
                rng = env.unwrapped.np_random
                # New-style NumPy Generator has no .seed method
                if isinstance(rng, np.random.Generator):
                    env.reset(seed=seed)
                else:
                    rng.seed(seed)
            else:
                # Fallback: reset environment with seed
                env.reset(seed=seed)
    
    def rollout(
        self,
        eps: float = 1e-10,
        speed_scale: float = 16.0,
        compute_log_prob: bool = True,
        return_sensitivity: bool = False
    ) -> dict:
        """Run one episode with noise injection and compute log-probability.
        
        Args:
            eps: Small epsilon for numerical stability in log computations.
            speed_scale: Reference speed for avg_speed_ratio normalization.
            compute_log_prob: Whether to compute log-probabilities (slower). 
                Set False for faster rollouts when only metrics are needed.
        
        Returns:
            Dictionary containing:
                - log_prob: Log-probability (0.0 if compute_log_prob=False)
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
            p_noise = float(torch.clamp(p_noise, 0.0, 1.0).item())
        else:
            p_noise = float(p_noise[0])

        # Sample observation noise using numpy (faster than torch for non-gradient ops)
        num_cars = 4
        noise_position_x = np.array(
            [float(sample_gaussian_mixture(env_params.initial_position_x, generator = self.torch_rng)) for _ in range(num_cars)]
        )
        noise_position_y = np.array(
            [float(sample_gaussian_mixture(env_params.initial_position_y, generator = self.torch_rng)) for _ in range(num_cars)]
        )
        noise_velocity_x = np.array(
            [float(sample_gaussian_mixture(env_params.velocity_x, generator = self.torch_rng)) for _ in range(num_cars)]
        )
        noise_velocity_y = np.array(
            [float(sample_gaussian_mixture(env_params.velocity_y, generator = self.torch_rng)) for _ in range(num_cars)]
        )

        # Pre-compute log-prob for initial state (computed once, not per-step)
        if compute_log_prob:
            log_prob, nominal_log_prob = self._compute_initial_log_prob(
                env, env_params, eps
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
        
        # For batched log-prob computation
        action_probs = []
        obs_noise_values = []

        unwrapped = env.unwrapped
        ego = unwrapped.vehicle
        policy_freq = unwrapped.config.get("policy_frequency", 1)
        
        passed_actions = []
        passed_available = []

        while not (done or truncated):
            # Add sensor noise (vectorized)
            obs_noisy = obs.copy()
            obs_noisy[1:num_cars+1, 1] += noise_position_x
            obs_noisy[1:num_cars+1, 2] += noise_position_y
            obs_noisy[1:num_cars+1, 3] += noise_velocity_x
            obs_noisy[1:num_cars+1, 4] += noise_velocity_y

            action, _states = model.predict(obs_noisy, deterministic=True)

            # High-level action fuzzing
            if return_sensitivity:
                passed_available.append(unwrapped.action_type.get_available_actions())
            if np.random.random() < p_noise:
                available = unwrapped.action_type.get_available_actions()
                action = unwrapped.np_random.choice(available)
                action_prob = p_noise / len(available)
            else:
                action_prob = 1.0 - p_noise
                
            if return_sensitivity:
                passed_actions.append(action)

            if compute_log_prob:
                action_probs.append(action_prob)
                obs_noise_values.append(obs_noisy[1:num_cars+1, 1:5].copy())

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
                    if d < min_dist:
                        min_dist = d

            if ego.on_road:
                on_road_steps += 1

            a = int(np.asarray(action).flat[0])
            if prev_action is not None and a != prev_action and a in [0, 2]:
                lane_changes += 1
            prev_action = a

        # Batch compute log-prob for actions and observations
        if compute_log_prob and action_probs:
            log_prob += self._compute_trajectory_log_prob(
                action_probs, obs_noise_values, env_params, eps
            )
            nominal_log_prob += self._compute_trajectory_log_prob(
                [1 - NOMINAL.high_lvl_ctrl_noise.p[0]] * len(action_probs),
                obs_noise_values, NOMINAL, eps
            )

        # Compute trajectory metrics
        velocities = np.array(velocities)
        dt = 1.0 / policy_freq
        accel = np.diff(velocities) / dt
        jerk = np.diff(accel) / dt

        decel = np.clip(-accel, 0, None)
        n = len(decel)
        brake_severity = float(np.sum(decel**2) / n) if n > 0 else 0.0

        mean_vel = float(np.mean(velocities)) if len(velocities) > 0 else 0.0
        jerk_score = float(np.mean(jerk**2)) if len(jerk) > 0 else 0.0
        on_road_frac = on_road_steps / max(total_steps, 1)

        trajectory_metrics = {
            "success": 0.0 if unwrapped.vehicle.crashed else 1.0,
            "min_distance": float(min_dist) if min_dist != float("inf") else 0.0,
            "avg_speed_ratio": mean_vel / speed_scale,
            "jerk_score": jerk_score,
            "brake_severity": brake_severity,
            "lane_changes": lane_changes,
            "on_road_frac": on_road_frac,
            "cumulative_reward": cumulative_reward,
        }
        
        # also return realized scenario params
        realized_params = {
            "initial_position_x": noise_position_x, 
            "initial_position_y": noise_position_y, 
            "velocity_x": noise_velocity_x, 
            "velocity_y": noise_velocity_y,
            "high_lvl_actions": {"actions": passed_actions, "available": passed_available},
            "initial_speed": env.unwrapped.vehicle.speed,
            "initial_heading": ["exr", "nxr", "wxr", "sxr"][env.unwrapped.road.vehicles[0].heading_idx],
            "politeness": [env.unwrapped.road.vehicles[i].POLITENESS for i in range(1,5)],
            "other_vehicles_speed": [env.unwrapped.road.vehicles[i].speed for i in range(1, 5)],
            "entering_vehicle_position": env.unwrapped.road.vehicles[-1].position[0], 
    
            
        }

        env.close()
        
        

        return {
            "log_prob": float(log_prob) if not isinstance(log_prob, float) else log_prob,
            "nominal_log_prob": float(nominal_log_prob) if not isinstance(nominal_log_prob, float) else nominal_log_prob,
            "is_failure": trajectory_metrics["success"] < 1e-2,
            "robustness": compute_robustness(trajectory_metrics, weights=self.robustness_weights),
            "metrics": trajectory_metrics,
        } if not return_sensitivity else {
            "log_prob": float(log_prob) if not isinstance(log_prob, float) else log_prob,
            "nominal_log_prob": float(nominal_log_prob) if not isinstance(nominal_log_prob, float) else nominal_log_prob,
            "is_failure": trajectory_metrics["success"] < 1e-2,
            "robustness": compute_robustness(trajectory_metrics, weights=self.robustness_weights),
            "metrics": trajectory_metrics,
            "realized_params": realized_params
        }
    
    def replay_rollout(
        self,
        obs_noise: dict,
        action_override: Optional[Dict[int, int]] = None,
        seed: Optional[int] = None,
        speed_scale: float = 16.0,
    ) -> dict:
        """Re-run a trajectory with fixed observation noise and optional action overrides.

        Used by per-timestep sensitivity analysis (Algorithm 11.1) and
        Shapley values (Algorithm 11.4). All randomness is controlled via
        the provided noise and overrides — no stochastic action fuzzing.

        Args:
            obs_noise: Dict with 'position_x', 'position_y', 'velocity_x',
                'velocity_y' (each np.ndarray of shape (4,)).
            action_override: Optional mapping {timestep: action_int}. At
                non-overridden steps the policy chooses deterministically.
            seed: Env reset seed for reproducible initial conditions.
            speed_scale: Reference speed for avg_speed_ratio.

        Returns:
            Dict with robustness, is_failure, metrics, actions.
        """
        env = self.env
        model = self.model
        if action_override is None:
            action_override = {}

        num_cars = 4
        noise_px = obs_noise["position_x"]
        noise_py = obs_noise["position_y"]
        noise_vx = obs_noise["velocity_x"]
        noise_vy = obs_noise["velocity_y"]

        done = truncated = False
        if seed is not None:
            obs, info = env.reset(seed=seed)
        else:
            obs, info = env.reset()

        velocities: List[float] = []
        min_dist = float("inf")
        lane_changes = 0
        on_road_steps = 0
        total_steps = 0
        cumulative_reward = 0.0
        prev_action = None
        actions_taken: List[int] = []

        unwrapped = env.unwrapped
        ego = unwrapped.vehicle
        policy_freq = unwrapped.config.get("policy_frequency", 1)

        t = 0
        while not (done or truncated):
            obs_noisy = obs.copy()
            obs_noisy[1:num_cars + 1, 1] += noise_px
            obs_noisy[1:num_cars + 1, 2] += noise_py
            obs_noisy[1:num_cars + 1, 3] += noise_vx
            obs_noisy[1:num_cars + 1, 4] += noise_vy

            if t in action_override:
                action = action_override[t]
            else:
                action, _ = model.predict(obs_noisy, deterministic=True)

            actions_taken.append(int(np.asarray(action).flat[0]))

            obs, reward, done, truncated, info = env.step(action)

            cumulative_reward += float(reward)
            total_steps += 1
            ego = unwrapped.vehicle
            v = np.sqrt(ego.velocity[0] ** 2 + ego.velocity[1] ** 2)
            velocities.append(v)

            for vehicle in unwrapped.road.vehicles:
                if vehicle is not ego:
                    d = np.linalg.norm(ego.position - vehicle.position)
                    if d < min_dist:
                        min_dist = d

            if ego.on_road:
                on_road_steps += 1

            a = actions_taken[-1]
            if prev_action is not None and a != prev_action and a in [0, 2]:
                lane_changes += 1
            prev_action = a
            t += 1

        velocities_arr = np.array(velocities)
        dt = 1.0 / policy_freq
        accel = np.diff(velocities_arr) / dt
        jerk = np.diff(accel) / dt

        decel = np.clip(-accel, 0, None)
        n_dec = len(decel)
        brake_severity = float(np.sum(decel ** 2) / n_dec) if n_dec > 0 else 0.0
        mean_vel = float(np.mean(velocities_arr)) if len(velocities_arr) > 0 else 0.0
        jerk_score = float(np.mean(jerk ** 2)) if len(jerk) > 0 else 0.0
        on_road_frac = on_road_steps / max(total_steps, 1)

        trajectory_metrics = {
            "success": 0.0 if unwrapped.vehicle.crashed else 1.0,
            "min_distance": float(min_dist) if min_dist != float("inf") else 0.0,
            "avg_speed_ratio": mean_vel / speed_scale,
            "jerk_score": jerk_score,
            "brake_severity": brake_severity,
            "lane_changes": lane_changes,
            "on_road_frac": on_road_frac,
            "cumulative_reward": cumulative_reward,
        }

        env.close()

        return {
            "is_failure": trajectory_metrics["success"] < 1e-2,
            "robustness": compute_robustness(trajectory_metrics, weights=self.robustness_weights),
            "metrics": trajectory_metrics,
            "actions": actions_taken,
        }

    def _compute_initial_log_prob(self, env, env_params, eps: float):
        """Compute log-prob for initial environment state (vectorized)."""
        log_prob = 0.0
        nominal_log_prob = 0.0
        
        # Vehicle speeds (batch)
        speeds = np.array([env.unwrapped.road.vehicles[i].speed for i in range(1, 5)])
        
        # Speed log-prob using numpy (faster than torch for non-gradient)
        mu, sigma = float(env_params.other_vehicle_speed.mu), float(env_params.other_vehicle_speed.sigma)
        log_prob += np.sum(-0.5 * ((speeds - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi)))
        
        nom_mu, nom_sigma = float(NOMINAL.other_vehicle_speed.mu), float(NOMINAL.other_vehicle_speed.sigma)
        nominal_log_prob += np.sum(-0.5 * ((speeds - nom_mu) / nom_sigma) ** 2 - np.log(nom_sigma * np.sqrt(2 * np.pi)))
        
        # Politeness log-prob (Beta distribution)
        pol_ab = env_params.politeness.ab
        alpha, beta_param = float(pol_ab[0]), float(pol_ab[1])
        for i in range(1, 5):
            pol = env.unwrapped.road.vehicles[i].POLITENESS
            if 0 < pol < 1:
                log_prob += (alpha - 1) * np.log(pol) + (beta_param  - 1) * np.log(1 - pol)
        
        nom_ab = NOMINAL.politeness.ab
        nom_alpha, nom_beta = float(nom_ab[0]), float(nom_ab[1])
        for i in range(1, 5):
            pol = env.unwrapped.road.vehicles[i].POLITENESS
            if 0 < pol < 1:
                nominal_log_prob += (nom_alpha - 1) * np.log(pol) + (nom_beta - 1) * np.log(1 - pol)
        
        # Entering vehicle position
        log_prob += float(torch.log(torch.clamp(
            gaussian_mixture_pdf(env.unwrapped.road.vehicles[-1].position[0], env_params.entering_vehicle_position) + eps, eps
        )))
        nominal_log_prob += float(torch.log(torch.clamp(
            gaussian_mixture_pdf(env.unwrapped.road.vehicles[-1].position[0], NOMINAL.entering_vehicle_position) + eps, eps
        )))
        
        # Ego initial speed
        ego_speed = env.unwrapped.vehicle.speed
        mu, sigma = float(env_params.initial_speed.mu), float(env_params.initial_speed.sigma)
        log_prob += -0.5 * ((ego_speed - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        
        nom_mu, nom_sigma = float(NOMINAL.initial_speed.mu), float(NOMINAL.initial_speed.sigma)
        nominal_log_prob += -0.5 * ((ego_speed - nom_mu) / nom_sigma) ** 2 - np.log(nom_sigma * np.sqrt(2 * np.pi))
        
        # Ego heading
        head_probs = np.array(env_params.initial_heading.p, dtype=np.float32)
        probs = np.append(head_probs, 1.0 - head_probs.sum())
        heading_idx = env.unwrapped.road.vehicles[0].heading_idx
        log_prob += np.log(probs[heading_idx] + eps)
        
        nom_head_probs = np.array(NOMINAL.initial_heading.p, dtype=np.float32)
        nom_probs = np.append(nom_head_probs, 1.0 - nom_head_probs.sum())
        nominal_log_prob += np.log(nom_probs[heading_idx] + eps)
        
        return log_prob, nominal_log_prob
    
    def _compute_trajectory_log_prob(self, action_probs, obs_noise_values, params, eps: float):
        """Compute log-prob for trajectory (batched)."""
        log_prob = 0.0
        
        # Action log-probs
        action_probs = np.array(action_probs)
        log_prob += np.sum(np.log(np.clip(action_probs, eps, None)))
        
        # Observation noise log-probs (batched over timesteps)
        for obs_noise in obs_noise_values:
            # obs_noise shape: (num_cars, 4) for pos_x, pos_y, vel_x, vel_y
            log_prob += float(torch.log(torch.clamp(
                gaussian_mixture_pdf(obs_noise[0, 0], params.initial_position_x) + eps, eps)))
            log_prob += float(torch.log(torch.clamp(
                gaussian_mixture_pdf(obs_noise[0, 1], params.initial_position_y) + eps, eps)))
            log_prob += float(torch.log(torch.clamp(
                gaussian_mixture_pdf(obs_noise[0, 2], params.velocity_x) + eps, eps)))
            log_prob += float(torch.log(torch.clamp(
                gaussian_mixture_pdf(obs_noise[0, 3], params.velocity_y) + eps, eps)))
        
        return log_prob
    
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
