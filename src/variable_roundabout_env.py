"""
Roundabout environment with configurable traffic vehicle count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium.envs.registration import register
from highway_env import utils
from highway_env.envs.roundabout_env import RoundaboutEnv


try:
    register(
        id="VariableRoundabout-v0",
        entry_point="src.variable_roundabout_env:VariableRoundaboutEnv",
    )
except Exception:
    pass  # Already registered


@dataclass(frozen=True)
class SpawnSlot:
    lane_index: tuple[str, str, int]
    longitudinal: float
    destinations: tuple[str, ...] = ("exr", "nxr", "sxr", "wxr")


class VariableRoundaboutEnv(RoundaboutEnv):
    """
    RoundaboutEnv variant that can spawn a configurable number of traffic vehicles.

    The observation defaults match scripts/sb3_roundabout_dqn_gnn.py: a padded
    Kinematics table with 12 rows and 5 features. That keeps the environment
    compatible with a GNN DQN trained against that fixed observation space.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "Kinematics",
                    "vehicles_count": 12,
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "absolute": True,
                    "features_range": {
                        "x": [-100, 100],
                        "y": [-100, 100],
                        "vx": [-15, 15],
                        "vy": [-15, 15],
                    },
                },
                "traffic_vehicles_count": 8,
                "traffic_vehicle_speed": 16.0,
                "traffic_vehicle_speed_deviation": 2.0,
                "traffic_vehicle_position_deviation": 1.0,
            }
        )
        return config

    def _make_vehicles(self) -> None:
        position_deviation = self.config["traffic_vehicle_position_deviation"]
        speed = self.config["traffic_vehicle_speed"]
        speed_deviation = self.config["traffic_vehicle_speed_deviation"]

        # Ego vehicle, matching highway_env's stock roundabout setup.
        ego_lane = self.road.network.get_lane(("ser", "ses", 0))
        ego_vehicle = self.action_type.vehicle_class(
            self.road,
            ego_lane.position(125.0, 0.0),
            speed=8.0,
            heading=ego_lane.heading_at(140.0),
        )
        try:
            ego_vehicle.plan_route_to("nxs")
        except AttributeError:
            pass
        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])
        spawn_slots = self._traffic_spawn_slots()
        traffic_count = int(self.config["traffic_vehicles_count"])

        if traffic_count > len(spawn_slots):
            raise ValueError(
                "traffic_vehicles_count exceeds available spawn slots "
                f"({traffic_count} > {len(spawn_slots)})."
            )

        for slot in spawn_slots[:traffic_count]:
            lane = self.road.network.get_lane(slot.lane_index)
            longitudinal = slot.longitudinal + self.np_random.normal() * position_deviation
            longitudinal = float(np.clip(longitudinal, 0.0, lane.length))
            vehicle_speed = speed + self.np_random.normal() * speed_deviation

            vehicle = other_vehicles_type.make_on_lane(
                self.road,
                slot.lane_index,
                longitudinal=longitudinal,
                speed=vehicle_speed,
            )
            vehicle.plan_route_to(self.np_random.choice(slot.destinations))
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

    @staticmethod
    def _traffic_spawn_slots() -> list[SpawnSlot]:
        # The first four slots mirror the stock roundabout traffic pattern.
        slots = [
            SpawnSlot(("we", "sx", 1), 5.0, ("exr", "nxr", "sxr")),
            SpawnSlot(("we", "sx", 0), 14.0, ("exr", "nxr", "sxr")),
            SpawnSlot(("sx", "se", 0), 8.0, ("exr", "nxr", "sxr")),
            SpawnSlot(("eer", "ees", 0), 50.0, ("exr", "nxr", "sxr")),
        ]

        # Extra entry-road traffic gives more variable approaching agents.
        for lane_index in [
            ("eer", "ees", 0),
            ("ner", "nes", 0),
            ("wer", "wes", 0),
        ]:
            for longitudinal in [25.0, 85.0, 115.0]:
                slots.append(SpawnSlot(lane_index, longitudinal))

        # Extra in-roundabout traffic covers both circular lanes.
        circular_segments = [
            ("se", "ex"),
            ("ex", "ee"),
            ("ee", "nx"),
            ("nx", "ne"),
            ("ne", "wx"),
            ("wx", "we"),
            ("we", "sx"),
            ("sx", "se"),
        ]
        for start, end in circular_segments:
            for lane_id in [0, 1]:
                slots.append(SpawnSlot((start, end, lane_id), 8.0))

        return slots
