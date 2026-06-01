"""Record multi-car roundabout rollouts with four-action BRT safety filters.

The nominal controller is the trained DQN-GNN policy. At every policy step,
the TwoCar8D value function is queried against each traffic vehicle and the
minimum value is used as the safety monitor. The safety wrapper can either use
the least-restrictive BRT filter or a finite-difference smooth blending filter.

When the filter intervenes, it chooses among four safety actions:
forward, stop, left, and right. Forward/left/right map to highway-env meta
actions, while stop preserves the existing hard-stop behavior.
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

# import brt_utils as bu  # noqa: E402 circular model
import brt_utils_body_geometry as bu
import src  # registers VariableRoundabout-v0  # noqa: E402


TRAFFIC_VEHICLES = 7
ROLLOUT_SEEDS = range(10)
# Monitor BRT more often than highway-env's default once-per-second policy rate.
SIMULATION_FREQUENCY = 15
POLICY_FREQUENCY = 5
SMOOTH_GAMMA_DEFAULT = 5.0

FILTER_LEAST_RESTRICTIVE = "least-restrictive"
FILTER_SMOOTH = "smooth"
SAFETY_ACTION_NAMES = ("FORWARD", "STOP", "LEFT", "RIGHT")


@dataclass(frozen=True)
class Prediction:
    value: float
    ego_state: np.ndarray
    crashed: bool


@dataclass(frozen=True)
class CandidateResult:
    name: str
    action: int
    prediction: Prediction
    vdot: float
    smooth_constraint: float
    distance_to_nominal: float


@dataclass(frozen=True)
class ControllerDecision:
    action: int
    mode: str
    selected_action_name: str | None
    candidate_results: list[CandidateResult]
    nominal_prediction: Prediction
    nominal_smooth_constraint: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize four-action BRT safety filters on roundabout rollouts."
    )
    parser.add_argument(
        "--filter",
        choices=[FILTER_LEAST_RESTRICTIVE, FILTER_SMOOTH],
        default=FILTER_LEAST_RESTRICTIVE,
        help=f"Safety filter to use. Default: {FILTER_LEAST_RESTRICTIVE}.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=SMOOTH_GAMMA_DEFAULT,
        help=f"Smooth blending gamma. Used only by --filter smooth. Default: {SMOOTH_GAMMA_DEFAULT}.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=len(ROLLOUT_SEEDS),
        help=f"Number of fixed seeds to run from seed 0. Default: {len(ROLLOUT_SEEDS)}.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="First rollout seed. Default: 0.",
    )
    return parser.parse_args()


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


def min_brt_value_at_states(states, dynamics, vf_model) -> float:
    ego = states[0]
    others = states[1:]
    if not others:
        return float(bu.VALUE_CMAP_LIMIT)

    others_arr = np.stack([other.as_array() for other in others])
    values = bu.values_at_ego_state(
        dynamics,
        vf_model,
        ego.as_array(),
        others_arr,
    )
    return float(np.min(values))


def title_suffix(
    filter_name: str,
    mode: str,
    ego_value: float,
    dominating_idx: int | None,
) -> str:
    if dominating_idx is None:
        return f"\nV={ego_value:.2f} | {filter_name} | {mode} | no others"
    return f"\nV={ego_value:.2f} | {filter_name} | {mode} | min=#{dominating_idx}"


def capture_frame_data(
    env,
    states,
    dynamics,
    vf_model,
    px_grid: np.ndarray,
    py_grid: np.ndarray,
    filter_name: str,
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
    title_suffix_per_frame.append(title_suffix(filter_name, mode, ego_value, dominating_idx))
    return ego_value


def scalar_action(action: int | np.ndarray) -> int:
    return int(np.asarray(action).item())


def safety_action_indexes(env) -> dict[str, int]:
    action_indexes = env.unwrapped.action_type.actions_indexes
    return {
        "FORWARD": action_indexes["FASTER"],
        "STOP": action_indexes["IDLE"],
        "LEFT": action_indexes["LANE_LEFT"],
        "RIGHT": action_indexes["LANE_RIGHT"],
    }


def apply_action_to_env(env, action: int, action_name: str | None) -> tuple[Any, float, bool, bool, dict]:
    if action_name == "STOP":
        controlled_ego = env.unwrapped.vehicle
        controlled_ego.speed = 0.0
        controlled_ego.target_speed = 0.0
    return env.step(action)


def predict_one_step(
    env,
    action: int,
    action_name: str | None,
    dynamics,
    vf_model,
    saved_controller_state: tuple[float, int] | None,
) -> Prediction:
    env_copy = copy.deepcopy(env)
    if saved_controller_state is not None and action_name != "STOP":
        controlled_ego = env_copy.unwrapped.vehicle
        controlled_ego.target_speed, controlled_ego.speed_index = saved_controller_state

    _obs, _reward, _terminated, _truncated, info = apply_action_to_env(
        env_copy, action, action_name
    )
    next_states = src.extract_vehicle_states(env_copy)
    ego_state = next_states[0].as_array()
    value = min_brt_value_at_states(next_states, dynamics, vf_model)
    crashed = bool(info.get("crashed", False))
    env_copy.close()
    return Prediction(value=value, ego_state=ego_state, crashed=crashed)


def ego_state_distance(lhs: np.ndarray, rhs: np.ndarray) -> float:
    delta = np.asarray(lhs, dtype=np.float64) - np.asarray(rhs, dtype=np.float64)
    delta[2] = (delta[2] + math.pi) % (2.0 * math.pi) - math.pi
    return float(np.linalg.norm(delta))


def build_candidate_results(
    env,
    current_value: float,
    nominal_prediction: Prediction,
    dynamics,
    vf_model,
    saved_controller_state: tuple[float, int] | None,
) -> list[CandidateResult]:
    dt = 1.0 / POLICY_FREQUENCY
    actions = safety_action_indexes(env)
    results: list[CandidateResult] = []

    for name in SAFETY_ACTION_NAMES:
        prediction = predict_one_step(
            env,
            actions[name],
            name,
            dynamics,
            vf_model,
            saved_controller_state,
        )
        vdot = (prediction.value - current_value) / dt
        results.append(
            CandidateResult(
                name=name,
                action=actions[name],
                prediction=prediction,
                vdot=vdot,
                smooth_constraint=vdot,
                distance_to_nominal=ego_state_distance(
                    prediction.ego_state,
                    nominal_prediction.ego_state,
                ),
            )
        )

    return results


def with_smooth_constraints(
    candidates: list[CandidateResult],
    current_value: float,
    gamma: float,
) -> list[CandidateResult]:
    return [
        CandidateResult(
            name=candidate.name,
            action=candidate.action,
            prediction=candidate.prediction,
            vdot=candidate.vdot,
            smooth_constraint=candidate.vdot + gamma * current_value,
            distance_to_nominal=candidate.distance_to_nominal,
        )
        for candidate in candidates
    ]


def choose_least_restrictive_action(
    env,
    nominal_action: int,
    current_value: float,
    dynamics,
    vf_model,
    saved_controller_state: tuple[float, int] | None,
) -> ControllerDecision:
    nominal_prediction = predict_one_step(
        env,
        nominal_action,
        None,
        dynamics,
        vf_model,
        saved_controller_state,
    )
    candidate_results = build_candidate_results(
        env,
        current_value,
        nominal_prediction,
        dynamics,
        vf_model,
        saved_controller_state,
    )
    candidate_results = with_smooth_constraints(candidate_results, current_value, 0.0)

    if current_value >= 0.0:
        return ControllerDecision(
            action=nominal_action,
            mode="DQN",
            selected_action_name=None,
            candidate_results=candidate_results,
            nominal_prediction=nominal_prediction,
            nominal_smooth_constraint=0.0,
        )

    selected = max(
        candidate_results,
        key=lambda candidate: (candidate.prediction.value, candidate.vdot),
    )
    return ControllerDecision(
        action=selected.action,
        mode=f"SAFE:{selected.name}",
        selected_action_name=selected.name,
        candidate_results=candidate_results,
        nominal_prediction=nominal_prediction,
        nominal_smooth_constraint=0.0,
    )


def choose_smooth_action(
    env,
    nominal_action: int,
    current_value: float,
    gamma: float,
    dynamics,
    vf_model,
    saved_controller_state: tuple[float, int] | None,
) -> ControllerDecision:
    dt = 1.0 / POLICY_FREQUENCY
    nominal_prediction = predict_one_step(
        env,
        nominal_action,
        None,
        dynamics,
        vf_model,
        saved_controller_state,
    )
    nominal_vdot = (nominal_prediction.value - current_value) / dt
    nominal_smooth_constraint = nominal_vdot + gamma * current_value
    candidate_results = build_candidate_results(
        env,
        current_value,
        nominal_prediction,
        dynamics,
        vf_model,
        saved_controller_state,
    )
    candidate_results = with_smooth_constraints(candidate_results, current_value, gamma)

    if nominal_smooth_constraint >= 0.0:
        return ControllerDecision(
            action=nominal_action,
            mode="DQN",
            selected_action_name=None,
            candidate_results=candidate_results,
            nominal_prediction=nominal_prediction,
            nominal_smooth_constraint=nominal_smooth_constraint,
        )

    safe_candidates = [
        candidate for candidate in candidate_results if candidate.smooth_constraint >= 0.0
    ]
    if safe_candidates:
        selected = min(
            safe_candidates,
            key=lambda candidate: (
                candidate.distance_to_nominal,
                -candidate.prediction.value,
                -candidate.vdot,
            ),
        )
    else:
        selected = max(
            candidate_results,
            key=lambda candidate: (candidate.vdot, candidate.prediction.value),
        )

    return ControllerDecision(
        action=selected.action,
        mode=f"SAFE:{selected.name}",
        selected_action_name=selected.name,
        candidate_results=candidate_results,
        nominal_prediction=nominal_prediction,
        nominal_smooth_constraint=nominal_smooth_constraint,
    )


def choose_controller_action(
    env,
    policy,
    obs,
    current_value: float,
    filter_name: str,
    gamma: float,
    dynamics,
    vf_model,
    saved_controller_state: tuple[float, int] | None,
) -> ControllerDecision:
    nominal_action, _states = policy.predict(obs, deterministic=True)
    nominal_action = scalar_action(nominal_action)

    if filter_name == FILTER_LEAST_RESTRICTIVE:
        return choose_least_restrictive_action(
            env,
            nominal_action,
            current_value,
            dynamics,
            vf_model,
            saved_controller_state,
        )
    if filter_name == FILTER_SMOOTH:
        return choose_smooth_action(
            env,
            nominal_action,
            current_value,
            gamma,
            dynamics,
            vf_model,
            saved_controller_state,
        )
    raise ValueError(f"Unknown filter: {filter_name}")


def format_candidate_summary(candidates: list[CandidateResult]) -> str:
    return ", ".join(
        f"{candidate.name}:Vnext={candidate.prediction.value:.3f},"
        f"sdot={candidate.smooth_constraint:.3f}"
        for candidate in candidates
    )


def rollout_with_safety_controller(
    env,
    policy,
    dynamics,
    vf_model,
    px_grid: np.ndarray,
    py_grid: np.ndarray,
    seed: int,
    filter_name: str,
    gamma: float,
) -> dict:
    """Run one rollout with the selected BRT safety filter."""
    obs, _info = env.reset(seed=seed)

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
        decision = choose_controller_action(
            env,
            policy,
            obs,
            ego_value,
            filter_name,
            gamma,
            dynamics,
            vf_model,
            saved_controller_state,
        )

        ego = states[0]
        others = states[1:]
        rollout_frames.append(env.render())
        frames_values.append(values)
        ego_xy.append((ego.px, ego.py))
        others_xy_per_frame.append([(other.px, other.py) for other in others])
        title_suffix_per_frame.append(
            title_suffix(filter_name, decision.mode, ego_value, dominating_idx)
        )
        controller_modes.append(decision.mode)

        controlled_ego = env.unwrapped.vehicle
        if decision.selected_action_name == "STOP":
            if saved_controller_state is None:
                saved_controller_state = (
                    float(controlled_ego.target_speed),
                    int(controlled_ego.speed_index),
                )
        elif saved_controller_state is not None:
            controlled_ego.target_speed, controlled_ego.speed_index = saved_controller_state
            saved_controller_state = None

        if decision.mode != "DQN":
            print(
                f"  seed={seed:02d} step={step_count:02d}: "
                f"filter={filter_name} V={ego_value:.3f} -> {decision.mode}; "
                f"nominal_sdot={decision.nominal_smooth_constraint:.3f}; "
                f"{format_candidate_summary(decision.candidate_results)}"
            )

        obs, _reward, terminated, truncated, info = apply_action_to_env(
            env,
            decision.action,
            decision.selected_action_name,
        )
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
        filter_name,
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


def gamma_label(gamma: float) -> str:
    return f"{gamma:g}".replace("-", "neg").replace(".", "p")


def output_stem(filter_name: str, gamma: float, seed: int) -> str:
    if filter_name == FILTER_SMOOTH:
        return f"four_action_smooth_gamma_{gamma_label(gamma)}_seed_{seed:02d}"
    return f"four_action_least_restrictive_seed_{seed:02d}"


def main() -> None:
    args = parse_args()
    seeds = range(args.start_seed, args.start_seed + args.num_seeds)

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
        f"filter={args.filter}; gamma={args.gamma:g}."
    )

    try:
        for seed in seeds:
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

            print(
                f"Running four-action safety rollout for seed={seed:02d} "
                f"with filter={args.filter}..."
            )
            rollout = rollout_with_safety_controller(
                env,
                policy,
                dynamics,
                vf_model,
                px_grid,
                py_grid,
                seed,
                args.filter,
                args.gamma,
            )
            modes = rollout["controller_modes"]
            dqn_steps = modes.count("DQN")
            safe_steps = len(modes) - dqn_steps
            print(
                f"  captured {len(rollout['rollout_frames'])} frames; "
                f"DQN steps={dqn_steps}, safe steps={safe_steps}, "
                f"crashed={rollout['crashed']}"
            )
            output_path = bu.VIDEOS_DIR / f"{output_stem(args.filter, args.gamma, seed)}.mp4"
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
