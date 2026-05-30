"""Record multi-car roundabout rollouts with a BRT-triggered stop controller.

The nominal controller is the trained DQN-GNN policy. At every policy step,
the TwoCar8D value function is queried against each traffic vehicle and the
minimum value is used as the safety monitor. If the ego value falls below a
configurable positive margin, the safety controller immediately stops the ego
vehicle for that interval instead of applying the nominal policy action.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import brt_utils as bu  # noqa: E402
import src  # registers VariableRoundabout-v0  # noqa: E402


TRAFFIC_VEHICLES = 7
ROLLOUT_SEEDS = range(10)
# Monitor BRT more often than highway-env's default once-per-second policy rate.
SIMULATION_FREQUENCY = 15
POLICY_FREQUENCY = 15

# Switch before entering the zero sublevel set to account for discrete control
# timing and provide time for the simple stop override to take effect.
SAFETY_VALUE_THRESHOLD = 50.0


def evaluate_brt_frame(
    states,
    dynamics,
    vf_model,
    px_grid: np.ndarray,
    py_grid: np.ndarray,
) -> tuple[np.ndarray, float, int | None]:
    """Return the most-dangerous vehicle's BRT slice and ego value."""
    ego = states[0]
    others = states[1:]
    if not others:
        values = np.full(px_grid.shape, bu.VALUE_CMAP_LIMIT, dtype=np.float32)
        return values, float(bu.VALUE_CMAP_LIMIT), None

    others_arr = np.stack([other.as_array() for other in others])
    v_at_ego = bu.values_at_ego_state(
        dynamics,
        vf_model,
        ego.as_array(),
        others_arr,
    )
    dangerous_idx = int(np.argmin(v_at_ego))

    values = bu.query_slice_single(
        dynamics,
        vf_model,
        px_grid,
        py_grid,
        ego_psi=ego.psi,
        ego_v=ego.v,
        other=others_arr[dangerous_idx],
    )
    return values, float(v_at_ego[dangerous_idx]), dangerous_idx


def title_suffix(mode: str, ego_value: float, dominating_idx: int | None) -> str:
    if dominating_idx is None:
        return f"\nV={ego_value:.2f} | {mode} | no others"
    return f"\nV={ego_value:.2f} | {mode} | min=#{dominating_idx}"


def capture_frame_data(
    env,
    states,
    dynamics,
    vf_model,
    px_grid: np.ndarray,
    py_grid: np.ndarray,
    mode: str,
    rollout_frames: list[np.ndarray],
    frames_values: list[np.ndarray],
    ego_xy: list[tuple[float, float]],
    others_xy_per_frame: list[list[tuple[float, float]]],
    title_suffix_per_frame: list[str],
) -> float:
    """Capture one rendered frame and its BRT/controller visualization data."""
    values, ego_value, dominating_idx = evaluate_brt_frame(
        states, dynamics, vf_model, px_grid, py_grid
    )
    ego = states[0]
    others = states[1:]
    rollout_frames.append(env.render())
    frames_values.append(values)
    ego_xy.append((ego.px, ego.py))
    others_xy_per_frame.append([(other.px, other.py) for other in others])
    title_suffix_per_frame.append(title_suffix(mode, ego_value, dominating_idx))
    return ego_value


def rollout_with_safety_controller(
    env,
    policy,
    dynamics,
    vf_model,
    px_grid: np.ndarray,
    py_grid: np.ndarray,
    seed: int,
) -> dict:
    """Run one rollout, stopping ego whenever its BRT value crosses the safety margin."""
    obs, _info = env.reset(seed=seed)
    idle_action = env.unwrapped.action_type.actions_indexes["IDLE"]

    rollout_frames: list[np.ndarray] = []
    frames_values: list[np.ndarray] = []
    ego_xy: list[tuple[float, float]] = []
    others_xy_per_frame: list[list[tuple[float, float]]] = []
    title_suffix_per_frame: list[str] = []
    controller_modes: list[str] = []

    saved_controller_state: tuple[float, int] | None = None
    terminated = False
    truncated = False
    crashed = False
    step_count = 0

    while not (terminated or truncated):
        states = src.extract_vehicle_states(env)
        values, ego_value, dominating_idx = evaluate_brt_frame(
            states, dynamics, vf_model, px_grid, py_grid
        )
        mode = "STOP" if ego_value < SAFETY_VALUE_THRESHOLD else "DQN"

        ego = states[0]
        others = states[1:]
        rollout_frames.append(env.render())
        frames_values.append(values)
        ego_xy.append((ego.px, ego.py))
        others_xy_per_frame.append([(other.px, other.py) for other in others])
        title_suffix_per_frame.append(title_suffix(mode, ego_value, dominating_idx))
        controller_modes.append(mode)

        controlled_ego = env.unwrapped.vehicle
        if mode == "STOP":
            if saved_controller_state is None:
                saved_controller_state = (
                    float(controlled_ego.target_speed),
                    int(controlled_ego.speed_index),
                )
            controlled_ego.speed = 0.0
            controlled_ego.target_speed = 0.0
            action = idle_action
            print(
                f"  seed={seed:02d} step={step_count:02d}: "
                f"V_ego={ego_value:.3f} < {SAFETY_VALUE_THRESHOLD:g} -> STOP"
            )
        else:
            if saved_controller_state is not None:
                controlled_ego.target_speed, controlled_ego.speed_index = saved_controller_state
                saved_controller_state = None
            action, _states = policy.predict(obs, deterministic=True)

        obs, _reward, terminated, truncated, info = env.step(action)
        crashed = bool(info.get("crashed", crashed))
        step_count += 1

    terminal_states = src.extract_vehicle_states(env)
    capture_frame_data(
        env,
        terminal_states,
        dynamics,
        vf_model,
        px_grid,
        py_grid,
        "terminal",
        rollout_frames,
        frames_values,
        ego_xy,
        others_xy_per_frame,
        title_suffix_per_frame,
    )
    return {
        "rollout_frames": rollout_frames,
        "frames_values": frames_values,
        "ego_xy": ego_xy,
        "others_xy_per_frame": others_xy_per_frame,
        "title_suffix_per_frame": title_suffix_per_frame,
        "controller_modes": controller_modes,
        "crashed": crashed,
    }


def rollout_with_nominal_controller(env, policy, seed: int) -> dict:
    """Run one rendered episode using only the deterministic DQN action."""
    obs, _info = env.reset(seed=seed)
    rollout_frames: list[np.ndarray] = [env.render()]
    terminated = False
    truncated = False
    crashed = False
    step_count = 0

    while not (terminated or truncated):
        action, _states = policy.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        rollout_frames.append(env.render())
        crashed = bool(info.get("crashed", crashed))
        step_count += 1

    return {
        "rollout_frames": rollout_frames,
        "steps": step_count,
        "crashed": crashed,
    }


def main() -> None:
    print("Loading DeepReach BRT and DQN-GNN policy...")
    dynamics = bu.build_cpu_dynamics()
    vf_model = bu.load_value_network(dynamics)
    config = bu.make_roundabout_config(TRAFFIC_VEHICLES)
    config["simulation_frequency"] = SIMULATION_FREQUENCY
    config["policy_frequency"] = POLICY_FREQUENCY
    env = gym.make("VariableRoundabout-v0", render_mode="rgb_array", config=config)
    policy = DQN.load(str(bu.MODEL_PATH), env=env)
    px_axis, py_axis, px_grid, py_grid = bu.make_grid()
    bu.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Monitoring safety at {POLICY_FREQUENCY} Hz "
        f"with dynamics integrated at {SIMULATION_FREQUENCY} Hz; "
        f"STOP threshold is V < {SAFETY_VALUE_THRESHOLD:g}."
    )

    try:
        for seed in ROLLOUT_SEEDS:
            print(f"Running nominal DQN rollout for seed={seed:02d}...")
            nominal_rollout = rollout_with_nominal_controller(env, policy, seed)
            print(
                f"  captured {len(nominal_rollout['rollout_frames'])} frames; "
                f"crashed={nominal_rollout['crashed']}"
            )
            nominal_output_path = bu.VIDEOS_DIR / f"nominal_rollout_seed_{seed:02d}.mp4"
            bu.save_rollout_video(
                nominal_output_path,
                nominal_rollout["rollout_frames"],
            )

            print(f"Running safety-controlled rollout for seed={seed:02d}...")
            rollout = rollout_with_safety_controller(
                env, policy, dynamics, vf_model, px_grid, py_grid, seed
            )
            modes = rollout["controller_modes"]
            stop_steps = modes.count("STOP")
            dqn_steps = modes.count("DQN")
            print(
                f"  captured {len(rollout['rollout_frames'])} frames; "
                f"DQN steps={dqn_steps}, STOP steps={stop_steps}, "
                f"crashed={rollout['crashed']}"
            )
            output_path = bu.VIDEOS_DIR / f"safety_rollout_and_brt_seed_{seed:02d}.mp4"
            bu.save_sidebyside_video(
                output_path,
                rollout["rollout_frames"],
                px_axis,
                py_axis,
                rollout["frames_values"],
                rollout["ego_xy"],
                rollout["others_xy_per_frame"],
                len(rollout["frames_values"]),
                title_suffix_per_frame=rollout["title_suffix_per_frame"],
            )
    finally:
        env.close()

    print("Done.")


if __name__ == "__main__":
    main()
