"""
StanfordRoundabout source package.

This package provides tools for trajectory distribution fuzzing and robustness
evaluation of autonomous vehicle policies in the HighwayEnv roundabout environment.

Modules:
    scenario_params: Dataclasses for configurable scenario parameters
    distributions: Helper functions for sampling and PDF calculations
    vehicle: CustomPoliteVehicle with learnable politeness
    simulated_env: SimulatedEnv - roundabout environment with fuzzable parameters
    fuzzer: ScenarioFuzzer class with rollout and optimization methods
    robustness: Robustness score computation

Example usage:
    import src  # auto-registers SimulatedEnv-v0
    
    env = gym.make("SimulatedEnv-v0", render_mode="rgb_array", scenario_params=src.NOMINAL)
    fuzzer = src.ScenarioFuzzer(env, model)
    result = fuzzer.rollout()
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
from .simulated_env import SimulatedEnv
from .fuzzer import ScenarioFuzzer, FuzzerConfig, PARAM_NAMES
from .robustness import compute_robustness, trajectory_metrics_from_rollout, weights_from_vector
from .failure_probability import (
    estimate_failure_probability,
    importance_sampling_estimate,
    print_failure_probability_report,
)

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
    # Fuzzer
    "ScenarioFuzzer",
    "FuzzerConfig",
    "PARAM_NAMES",
    # Robustness
    "compute_robustness",
    "trajectory_metrics_from_rollout",
    "weights_from_vector",
    # Failure probability estimation
    "estimate_failure_probability",
    "importance_sampling_estimate",
    "print_failure_probability_report",
]
