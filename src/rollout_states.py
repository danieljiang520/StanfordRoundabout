"""Per-vehicle state extraction for highway-env rollouts.

Produces absolute-frame (px, py, psi, v) for the ego and every other vehicle
at each timestep, suitable for feeding into a TwoCar8D BRT value function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VehicleState:
    px: float
    py: float
    psi: float
    v: float

    def as_array(self) -> np.ndarray:
        return np.array([self.px, self.py, self.psi, self.v], dtype=np.float64)


def _vehicle_to_state(vehicle) -> VehicleState:
    return VehicleState(
        px=float(vehicle.position[0]),
        py=float(vehicle.position[1]),
        psi=float(vehicle.heading),
        v=float(vehicle.speed),
    )


def extract_vehicle_states(env) -> list[VehicleState]:
    """Return [ego_state, other1_state, ...] in absolute world frame.

    Ego is placed first explicitly via env.unwrapped.vehicle. The remaining
    vehicles preserve their order in env.unwrapped.road.vehicles.
    """
    unwrapped = env.unwrapped
    ego = unwrapped.vehicle
    states = [_vehicle_to_state(ego)]
    for vehicle in unwrapped.road.vehicles:
        if vehicle is ego:
            continue
        states.append(_vehicle_to_state(vehicle))
    return states


def rollout_with_states(
    env,
    model,
    deterministic: bool = True,
    max_steps: int | None = None,
    seed: int | None = None,
    capture_frames: bool = False,
) -> dict:
    """Run one episode, recording per-vehicle states at every timestep.

    states[i] is the env state right before action actions[i] is applied.
    The list has length T+1 — one extra entry for the post-terminal state.
    If capture_frames is True, also records env.render() output per timestep
    under "frames" (requires env to be made with render_mode="rgb_array").
    """
    obs, _info = env.reset(seed=seed) if seed is not None else env.reset()

    states = [extract_vehicle_states(env)]
    actions: list[int] = []
    rewards: list[float] = []
    frames: list[np.ndarray] = [env.render()] if capture_frames else []

    terminated = False
    truncated = False
    crashed = False
    step_count = 0

    while not (terminated or truncated):
        if max_steps is not None and step_count >= max_steps:
            break

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        actions.append(int(np.asarray(action).item()))
        rewards.append(float(reward))
        states.append(extract_vehicle_states(env))
        if capture_frames:
            frames.append(env.render())
        crashed = bool(info.get("crashed", crashed))
        step_count += 1

    result = {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
        "crashed": crashed,
    }
    if capture_frames:
        result["frames"] = frames
    return result
