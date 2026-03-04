"""
Fuzzer for finding critical failure scenarios in trajectory simulation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from .scenario_params import SQRT_2, ScenarioParams
from .rollout import rollout


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
        >>> result = fuzzer.optimize()
        >>> print(f"Critical params: {fuzzer.get_params_dict(result.x)}")
    """
    
    def __init__(
        self,
        env,
        model,
        scenario_params: ScenarioParams,
        robustness_weights: Optional[Dict[str, float]] = None,
        config: Optional[FuzzerConfig] = None,
    ):
        """Initialize the fuzzer.
        
        Args:
            env: Gymnasium environment with scenario_params attribute.
            model: Trained policy model (e.g., DQN).
            scenario_params: ScenarioParams object to modify during fuzzing.
            robustness_weights: Optional weights for robustness computation.
            config: FuzzerConfig with optimization settings. Uses defaults if None.
        """
        self.env = env
        self.model = model
        self.scenario_params = scenario_params
        self.robustness_weights = robustness_weights
        self.config = config or FuzzerConfig()
        self._eval_count = 0
        self._history: List[Dict[str, Any]] = []
    
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
        
        rollout_kwargs = (
            {"robustness_weights": self.robustness_weights}
            if self.robustness_weights else {}
        )
        results = [
            rollout(self.env, self.model, **rollout_kwargs)
            for _ in range(self.config.n_samples)
        ]
        
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
        """Convert parameter vector to named dictionary.
        
        Args:
            x: Parameter vector of length 10.
        
        Returns:
            Dictionary mapping parameter names to values.
        """
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
