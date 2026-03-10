"""
Custom vehicle class with learnable politeness parameter.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
from highway_env.road.road import Road
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.objects import RoadObject

from .scenario_params import BetaParam

LaneIndex = Tuple[str, str, int]


class CustomPoliteVehicle(IDMVehicle):
    """IDMVehicle with a politeness parameter drawn from a Beta distribution.
    
    The politeness parameter affects lane-changing behavior in the IDM model.
    It is sampled from Beta(alpha, beta) where:
        alpha = politeness[0]
        beta = alpha + politeness[1]
    
    This allows for differentiable sampling via rsample().
    """

    def __init__(
        self,
        road: Road,
        position,
        heading: float,
        speed: float,
        politeness: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Initialize the vehicle.
        
        Args:
            road: The road the vehicle is on.
            position: Initial position [x, y].
            heading: Initial heading angle (radians).
            speed: Initial speed (m/s).
            politeness: Optional [alpha, beta-alpha] for Beta distribution.
                        Defaults to [1.0, 1.0] (uniform).
            **kwargs: Additional arguments passed to IDMVehicle.
        """
        super().__init__(road, position, heading, speed, **kwargs)

        if politeness is None:
            politeness = torch.tensor([1.0, 1.0], dtype=torch.float)

        # Ensure tensor
        if isinstance(politeness, nn.Parameter):
            tensor_p = politeness
        else:
            tensor_p = torch.tensor(politeness, dtype=torch.float)

        alpha = tensor_p[0]
        beta = tensor_p[1]

        # Differentiable sample
        dist = torch.distributions.Beta(alpha, beta)
        self.POLITENESS = dist.rsample()

    @classmethod
    def make_on_lane(
        cls,
        road: Road,
        lane_index: LaneIndex,
        longitudinal: float,
        speed: Optional[float] = None,
        politeness: Optional[BetaParam] = None,
    ) -> RoadObject:
        """Create a vehicle on a given lane at a longitudinal position.
        
        Args:
            road: Road object containing the road network.
            lane_index: Index of the lane (from, to, id).
            longitudinal: Longitudinal position along the lane (meters).
            speed: Initial speed (m/s). Defaults to lane speed limit.
            politeness: Optional BetaParam for politeness distribution.
        
        Returns:
            A CustomPoliteVehicle at the specified position.
        """
        lane = road.network.get_lane(lane_index)
        if speed is None:
            speed = lane.speed_limit
        return cls(
            road,
            lane.position(longitudinal, 0),
            lane.heading_at(longitudinal),
            speed,
            politeness,
        )
