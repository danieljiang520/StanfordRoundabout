"""
Scenario parameter dataclasses for trajectory distribution fuzzing.

These dataclasses encapsulate learnable/fuzzable parameters for observation,
action, and environment disturbances.
"""
import math
from dataclasses import dataclass
from typing import List, Union

import torch

ParamType = Union[float, int, List[float], torch.Tensor, torch.nn.Parameter]

SQRT_2 = math.sqrt(2)


@dataclass
class GaussianMixtureParam:
    """Mixture of n Gaussians.
    
    Attributes:
        p: Probabilities for the first n-1 components (last is 1 - sum(p)).
        mu: Means for each component (length n).
        sigma: Standard deviations for each component (length n).
    """
    p: ParamType
    mu: ParamType
    sigma: ParamType


@dataclass
class NormalParam:
    """Single Gaussian distribution.
    
    Attributes:
        mu: Mean.
        sigma: Standard deviation.
    """
    mu: ParamType
    sigma: ParamType


@dataclass
class ProbabilityParam:
    """Categorical probability distribution.
    
    Attributes:
        p: Probabilities (length n-1 for n categories, or length 1 for binary).
    """
    p: ParamType


@dataclass
class BetaParam:
    """Beta distribution parameters.
    
    Attributes:
        ab: Two-element [alpha, beta] for Beta(alpha, beta).
    """
    ab: ParamType


@dataclass
class ScenarioParams:
    """All scenario parameters for trajectory distribution fuzzing.
    
    Observation disturbances:
        initial_position_x, initial_position_y: Position noise (GaussianMixture).
        velocity_x, velocity_y: Velocity noise (GaussianMixture).
    
    Action disturbances:
        high_lvl_ctrl_noise: Probability of random action (Probability).
        initial_speed: Ego vehicle initial speed (Normal, clipped to [0, 16]).
        initial_heading: Ego heading probabilities for [east, north, west]
                         (south = 1 - sum).
    
    Environment disturbances:
        politeness: Other vehicles' politeness (Beta).
        other_vehicle_speed: Speed of other vehicles (Normal, clipped to [0, 32]).
        entering_vehicle_position: Longitudinal position of entering vehicle
                                   (GaussianMixture).
    """
    # Observation
    initial_position_x: GaussianMixtureParam
    initial_position_y: GaussianMixtureParam
    velocity_x: GaussianMixtureParam
    velocity_y: GaussianMixtureParam

    # Action
    high_lvl_ctrl_noise: ProbabilityParam
    initial_speed: NormalParam
    initial_heading: ProbabilityParam

    # Environment
    politeness: BetaParam
    other_vehicle_speed: NormalParam
    entering_vehicle_position: GaussianMixtureParam


# Default nominal scenario parameters
NOMINAL = ScenarioParams(
    # Observation disturbances (very small noise)
    initial_position_x=GaussianMixtureParam(
        p=[1.0],
        mu=[0.0, 0.0],
        sigma=[0.005, 0.005],
    ),
    initial_position_y=GaussianMixtureParam(
        p=[1.0],
        mu=[0.0, 0.0],
        sigma=[0.005, 0.005],
    ),
    velocity_x=GaussianMixtureParam(
        p=[1.0],
        mu=[0.0, 0.0],
        sigma=[SQRT_2 * 0.005 / 0.1, SQRT_2 * 0.005 / 0.1],
    ),
    velocity_y=GaussianMixtureParam(
        p=[1.0],
        mu=[0.0, 0.0],
        sigma=[SQRT_2 * 0.005 / 0.1, SQRT_2 * 0.005 / 0.1],
    ),
    # Action disturbances
    high_lvl_ctrl_noise=ProbabilityParam(p=[0.0]),
    initial_speed=NormalParam(mu=8.0, sigma=0.5),
    initial_heading=ProbabilityParam(p=[0.33, 0.33, 0.33]),
    # Environment disturbances
    politeness=BetaParam(ab=[1, 1]),  # uniform distribution
    other_vehicle_speed=NormalParam(mu=16.0, sigma=2.0),
    entering_vehicle_position=GaussianMixtureParam(
        p=[1.0],
        mu=[5.0, 5.0],
        sigma=[2.0, 2.0],
    ),
)
