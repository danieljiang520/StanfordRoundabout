"""
Simulated roundabout environment with configurable scenario parameters.
"""
from typing import Optional

import torch
from gymnasium.envs.registration import register
from highway_env.envs.roundabout_env import RoundaboutEnv
import numpy as np

from .scenario_params import NOMINAL, ScenarioParams
from .vehicle import CustomPoliteVehicle

# Register the environment when module is imported
try:
    register(
        id="SimulatedEnv-v0",
        entry_point="src.simulated_env:SimulatedEnv",
    )
except Exception:
    pass  # Already registered


class SimulatedEnv(RoundaboutEnv):
    """Roundabout environment with learnable/fuzzable scenario parameters.
    
    Extends the base RoundaboutEnv to support:
    - Configurable ego vehicle initial speed and heading
    - Configurable other vehicle speeds and politeness
    - Configurable entering vehicle position
    
    These parameters can be PyTorch tensors for gradient-based optimization.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        scenario_params: Optional[ScenarioParams] = None,
        render_mode: Optional[str] = None,
        seed: Optional[int] = 42
    ):
        """Initialize the simulated environment.
        
        Args:
            config: Optional environment configuration dictionary.
            scenario_params: Optional ScenarioParams for fuzzing. Defaults to NOMINAL.
            render_mode: Rendering mode ("human", "rgb_array", or None).
            seed: Random seed for reproducibility.
        """
        if scenario_params is None:
            scenario_params = NOMINAL
        
        self.scenario_params = scenario_params
        self._init_seed = seed
        self.torch_rng = torch.Generator()
        self._seed_rngs(seed)
        
        super().__init__(config=config, render_mode=render_mode)

    def _seed_rngs(self, seed: Optional[int]) -> None:
        """Seed all random number generators for reproducibility."""
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
            self.torch_rng.manual_seed(seed)
            torch.manual_seed(seed)
        else:
            self.np_random = np.random.default_rng()

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset the environment with optional seed for reproducibility.
        
        Args:
            seed: If provided, seeds all RNGs for reproducible episodes.
            options: Additional reset options.
        
        Returns:
            observation, info tuple
        """
        if seed is not None:
            self._seed_rngs(seed)
        elif self._init_seed is not None:
            # Re-seed with initial seed for consistent behavior across resets
            # Comment out this line if you want different episodes on each reset
            pass  # Don't re-seed by default to get varied episodes
        
        return super().reset(seed=seed, options=options)

    def _make_vehicles(self) -> None:
        """Create vehicles with scenario-dependent parameters."""
        position_deviation = 2.0
        min_speed, max_speed = 0.0, 32.0

        # Ego vehicle
        initial_mu = torch.as_tensor(self.scenario_params.initial_speed.mu)
        initial_sigma = torch.as_tensor(self.scenario_params.initial_speed.sigma)
        initial_speed = torch.clamp(
            initial_mu + initial_sigma * torch.randn(initial_mu.shape, generator = self.torch_rng),
            min=min_speed,
            max=16.0,
        )

        ego_lane = self.road.network.get_lane(("ser", "ses", 0))
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(125.0, 0.0),
            speed=float(initial_speed),
            heading=ego_lane.heading_at(140.0),
        )

        try:
            # Sample heading from categorical distribution
            p_tensor = torch.tensor(
                self.scenario_params.initial_heading.p, dtype=torch.float32
            )
            p_s = 1.0 - torch.sum(p_tensor)
            probs = torch.cat([p_tensor, p_s.unsqueeze(0)])

            # cat = torch.distributions.Categorical(probs) #comeback-1
            # heading_idx = cat.sample()
            heading_idx = torch.multinomial(
                probs,
                1,
                generator=self.torch_rng
            ).squeeze()
            

            destination = ["exr", "nxr", "wxr", "sxr"][heading_idx.item()]
            ego_vehicle.heading_idx = heading_idx
            ego_vehicle.plan_route_to(destination)
        except AttributeError:
            pass

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        # Other vehicles
        destinations = ["exr", "sxr", "nxr"]
        other_vehicles_type = CustomPoliteVehicle

        other_mu = torch.as_tensor(self.scenario_params.other_vehicle_speed.mu)
        other_sigma = torch.as_tensor(self.scenario_params.other_vehicle_speed.sigma)
        other_car_speed = torch.clamp(
            other_mu + other_sigma * torch.randn(other_mu.shape, generator = self.torch_rng),
            min=min_speed,
            max=max_speed,
        )

        # Incoming vehicle
        vehicle = other_vehicles_type.make_on_lane(
            road=self.road,
            lane_index=("we", "sx", 1),
            longitudinal=50.0 + self.np_random.normal() * position_deviation,
            speed=float(other_car_speed),
            politeness=self.scenario_params.politeness.ab,
        )

        if self.config["incoming_vehicle_destination"] is not None:
            destination = destinations[self.config["incoming_vehicle_destination"]]
        else:
            destination = self.np_random.choice(destinations)
        vehicle.plan_route_to(destination)
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)

        # Additional vehicles on inner lane
        for i in list(range(1, 2)) + list(range(-1, 0)):
            vehicle = other_vehicles_type.make_on_lane(
                road=self.road,
                lane_index=("we", "sx", 0),
                longitudinal=20.0 * float(i)
                + self.np_random.normal() * position_deviation,
                speed=float(other_car_speed),
                politeness=self.scenario_params.politeness.ab,
            )
            vehicle.plan_route_to(self.np_random.choice(destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        # Entering vehicle
        entering_position = 0
        last_prob = 1
        num_mixtures = len(self.scenario_params.entering_vehicle_position.mu)
        entering_pos_mu = torch.as_tensor(
            self.scenario_params.entering_vehicle_position.mu
        )
        entering_pos_sigma = torch.as_tensor(
            self.scenario_params.entering_vehicle_position.sigma
        )

        for idx in range(num_mixtures - 1):
            entering_position += self.scenario_params.entering_vehicle_position.p[idx] * (
                entering_pos_mu[idx]
                + entering_pos_sigma[idx] * torch.randn(entering_pos_mu[idx].shape, generator = self.torch_rng)
            )
            last_prob -= self.scenario_params.entering_vehicle_position.p[idx]

        entering_position += last_prob * (
            entering_pos_mu[num_mixtures - 1]
            + entering_pos_sigma[num_mixtures - 1]
            * torch.randn(entering_pos_mu[num_mixtures - 1].shape, generator = self.torch_rng)
        )

        vehicle = other_vehicles_type.make_on_lane(
            road=self.road,
            lane_index=("eer", "ees", 0),
            longitudinal=entering_position.item(),
            speed=float(other_car_speed),
            politeness=self.scenario_params.politeness.ab,
        )
        vehicle.plan_route_to(self.np_random.choice(destinations))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)



