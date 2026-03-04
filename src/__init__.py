"""
StanfordRoundabout source package.

This package provides tools for trajectory distribution fuzzing and robustness
evaluation of autonomous vehicle policies in the HighwayEnv roundabout environment.

Modules:
    scenario_params: Dataclasses for configurable scenario parameters
    distributions: Helper functions for sampling and PDF calculations
    vehicle: CustomPoliteVehicle with learnable politeness
    env: SimulatedEnv - roundabout environment with fuzzable parameters
    rollout: Rollout function with log-probability computation
    fuzzer: ScenarioFuzzer class for finding critical failure scenarios
    robustness: Robustness score computation

Example usage:
    from src.scenario_params import ScenarioParams, NOMINAL
    from src.env import SimulatedEnv, register_env
    from src.rollout import rollout
    from src.robustness import compute_robustness
    
    # Register the environment
    register_env()
    
    # Create environment with custom parameters
    env = gym.make("SimulatedEnv-v0", render_mode="rgb_array", scenario_params=NOMINAL)
"""

# Re-export commonly used items
from .scenario_params import (
    BetaParam,
    GaussianMixtureParam,
    NormalParam,
    ProbabilityParam,
    ScenarioParams,
    NOMINAL,
    SQRT_2,
)
from .distributions import (
    to_tensor,
    sample_gaussian_mixture,
    gaussian_pdf,
    gaussian_mixture_pdf,
)
from .vehicle import CustomPoliteVehicle
from .env import SimulatedEnv, register_env
from .rollout import rollout
from .fuzzer import ScenarioFuzzer, FuzzerConfig, PARAM_NAMES
from .robustness import compute_robustness, trajectory_metrics_from_rollout, weights_from_vector

__all__ = [
    # Scenario params
    "BetaParam",
    "GaussianMixtureParam",
    "NormalParam",
    "ProbabilityParam",
    "ScenarioParams",
    "NOMINAL",
    "SQRT_2",
    # Distributions
    "to_tensor",
    "sample_gaussian_mixture",
    "gaussian_pdf",
    "gaussian_mixture_pdf",
    # Vehicle
    "CustomPoliteVehicle",
    # Environment
    "SimulatedEnv",
    "register_env",
    # Rollout
    "rollout",
    # Fuzzer
    "ScenarioFuzzer",
    "FuzzerConfig",
    "PARAM_NAMES",
    # Robustness
    "compute_robustness",
    "trajectory_metrics_from_rollout",
    "weights_from_vector",
]
