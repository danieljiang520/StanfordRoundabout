"""Record multi-car roundabout rollouts with HJ optimal BRT safety control.

The nominal controller is the trained DQN-GNN policy. The safety monitor is the
minimum TwoCar8D BRT value over traffic vehicles. When safety intervenes, this
script applies continuous controls from the Hamilton-Jacobi safety problem
instead of highway-env discrete meta-actions.
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
import torch
from scipy.optimize import minimize
from stable_baselines3 import DQN

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import brt_utils_body_geometry as bu  # noqa: E402
import src  # registers VariableRoundabout-v0  # noqa: E402


TRAFFIC_VEHICLES = 1
ROLLOUT_SEEDS = range(10)
SIMULATION_FREQUENCY = 15
POLICY_FREQUENCY = 5
DEFAULT_FILTER = "least-restrictive"
DEFAULT_GAMMA = 1.0
DEFAULT_VALUE_THRESHOLD = 0
DEFAULT_LANE_STEERING_MARGIN = 0.25
DEFAULT_NUM_SEEDS = len(ROLLOUT_SEEDS)
DEFAULT_START_SEED = 0

FILTER_LEAST_RESTRICTIVE = "least-restrictive"
FILTER_SMOOTH = "smooth"
MODE_DQN = "DQN"
MODE_HJ_BANG = "HJ:BANG"
MODE_HJ_PROJECTED = "HJ:PROJECTED"
MODE_HJ_FALLBACK = "HJ:FALLBACK"


@dataclass(frozen=True)
class SafetyContext:
    value: float
    dangerous_idx: int | None
    state: torch.Tensor | None
    dvds: torch.Tensor | None


@dataclass(frozen=True)
class ControllerDecision:
    mode: str
    nominal_action: int | None
    control: np.ndarray | None
    smooth_constraint: float | None
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize HJ optimal BRT safety controllers on roundabout rollouts."
    )
    parser.add_argument(
        "--filter",
        choices=[FILTER_LEAST_RESTRICTIVE, FILTER_SMOOTH],
        default=DEFAULT_FILTER,
        help=f"Safety filter to use. Default: {DEFAULT_FILTER}.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help=f"Smooth blending gamma. Used only by --filter smooth. Default: {DEFAULT_GAMMA:g}.",
    )
    parser.add_argument(
        "--value-threshold",
        type=float,
        default=DEFAULT_VALUE_THRESHOLD,
        help=(
            "Least-restrictive filter intervention threshold. "
            f"Default: {DEFAULT_VALUE_THRESHOLD:g}."
        ),
    )
    parser.add_argument(
        "--lane-steering-margin",
        type=float,
        default=DEFAULT_LANE_STEERING_MARGIN,
        help=(
            "Maximum HJ steering deviation from highway-env lane-following steering, "
            f"in radians. Default: {DEFAULT_LANE_STEERING_MARGIN:g}."
        ),
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help=f"Number of fixed seeds to run from start seed. Default: {DEFAULT_NUM_SEEDS}.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=DEFAULT_START_SEED,
        help=f"First rollout seed. Default: {DEFAULT_START_SEED}.",
    )
    return parser.parse_args()


def scalar_action(action: int | np.ndarray) -> int:
    return int(np.asarray(action).item())


def make_state_tensor(ego_state: np.ndarray, others: np.ndarray) -> torch.Tensor:
    ego_t = torch.from_numpy(np.asarray(ego_state, dtype=np.float32))
    others_t = torch.from_numpy(np.asarray(others, dtype=np.float32))
    ego_block = ego_t.unsqueeze(0).expand(others_t.shape[0], 4)
    return torch.cat([ego_block, others_t], dim=-1)


def value_and_gradient_at_ego(states, dynamics, vf_model) -> SafetyContext:
    """Return min-over-vehicles V and dV/dx for the selected two-car state."""
    ego = states[0]
    others = states[1:]
    if not others:
        return SafetyContext(float(bu.VALUE_CMAP_LIMIT), None, None, None)

    others_arr = np.stack([other.as_array() for other in others])
    state = make_state_tensor(ego.as_array(), others_arr)
    t = torch.full((state.shape[0], 1), bu.T_QUERY, dtype=torch.float32)
    coords = torch.cat([t, state], dim=-1)
    model_input = dynamics.coord_to_input(coords)

    results = vf_model({"coords": model_input})
    model_in = results["model_in"]
    model_out = results["model_out"].squeeze(dim=-1)
    values = dynamics.io_to_value(model_in.detach(), model_out.detach())
    dvs = dynamics.io_to_dv(model_in, model_out).detach()[..., 1:]

    dangerous_idx = int(torch.argmin(values).item())
    return SafetyContext(
        value=float(values[dangerous_idx].item()),
        dangerous_idx=dangerous_idx,
        state=state[dangerous_idx : dangerous_idx + 1],
        dvds=dvs[dangerous_idx : dangerous_idx + 1],
    )


def min_brt_value_at_states(states, dynamics, vf_model) -> float:
    return value_and_gradient_at_ego(states, dynamics, vf_model).value


def evaluate_brt_frame(
    states,
    dynamics,
    vf_model,
    px_grid: np.ndarray,
    py_grid: np.ndarray,
) -> tuple[np.ndarray, SafetyContext]:
    ego = states[0]
    others = states[1:]
    context = value_and_gradient_at_ego(states, dynamics, vf_model)
    if context.dangerous_idx is None:
        values = np.full(px_grid.shape, bu.VALUE_CMAP_LIMIT, dtype=np.float32)
        return values, context

    others_arr = np.stack([other.as_array() for other in others])
    values = bu.query_slice_single(
        dynamics,
        vf_model,
        px_grid,
        py_grid,
        ego_psi=ego.psi,
        ego_v=ego.v,
        other=others_arr[context.dangerous_idx],
    )
    return values, context


def title_suffix(filter_name: str, decision: ControllerDecision, context: SafetyContext) -> str:
    controller = MODE_DQN if decision.mode == MODE_DQN else filter_name
    control = ""
    if decision.control is not None:
        control = f" | yaw={decision.control[0]:+.2f}, a={decision.control[1]:+.2f}"
    if context.dangerous_idx is None:
        return f"\nV={context.value:.2f} | {controller} | min=none{control}"
    return f"\nV={context.value:.2f} | {controller} | min=#{context.dangerous_idx}{control}"


def capture_frame_data(
    env,
    states,
    dynamics,
    vf_model,
    px_grid: np.ndarray,
    py_grid: np.ndarray,
    filter_name: str,
    decision: ControllerDecision,
    rollout_frames: list[np.ndarray],
    frames_values: list[np.ndarray],
    ego_xy: list[tuple[float, float]],
    others_xy_per_frame: list[list[tuple[float, float]]],
    title_suffix_per_frame: list[str],
) -> SafetyContext:
    values, context = evaluate_brt_frame(states, dynamics, vf_model, px_grid, py_grid)
    ego = states[0]
    others = states[1:]
    rollout_frames.append(env.render())
    frames_values.append(values)
    ego_xy.append((ego.px, ego.py))
    others_xy_per_frame.append([(other.px, other.py) for other in others])
    title_suffix_per_frame.append(title_suffix(filter_name, decision, context))
    return context


def raw_action_from_control(control: np.ndarray) -> dict[str, float]:
    return {"steering": float(control[0]), "acceleration": float(control[1])}


def step_with_continuous_ego_control(env, control: np.ndarray):
    """Step one policy interval while bypassing DiscreteMetaAction for ego."""
    from highway_env.vehicle.kinematics import Vehicle

    unwrapped = env.unwrapped
    raw_action = raw_action_from_control(control)
    frames = int(unwrapped.config["simulation_frequency"] // unwrapped.config["policy_frequency"])
    unwrapped.time += 1.0 / unwrapped.config["policy_frequency"]

    for _frame in range(frames):
        for vehicle in unwrapped.road.vehicles:
            if vehicle is not unwrapped.vehicle:
                vehicle.act()
        Vehicle.act(unwrapped.vehicle, raw_action)
        unwrapped.road.step(1.0 / unwrapped.config["simulation_frequency"])
        unwrapped.steps += 1

    obs = unwrapped.observation_type.observe()
    reward = unwrapped._reward(raw_action)
    terminated = unwrapped._is_terminated()
    truncated = unwrapped._is_truncated()
    info = unwrapped._info(obs, raw_action)
    if unwrapped.render_mode == "human":
        unwrapped.render()
    return obs, reward, terminated, truncated, info


def nominal_continuous_control(env, nominal_action: int) -> np.ndarray:
    env_copy = copy.deepcopy(env)
    env_copy.unwrapped.action_type.act(nominal_action)
    raw_action = env_copy.unwrapped.vehicle.action
    control = np.array(
        [float(raw_action["steering"]), float(raw_action["acceleration"])],
        dtype=np.float64,
    )
    env_copy.close()
    return clip_control(env.unwrapped, control)


def control_bounds(dynamics) -> list[tuple[float, float]]:
    return [(float(lo), float(hi)) for lo, hi in dynamics.control_range(None)]


def lane_following_steering(unwrapped_env) -> float:
    ego = unwrapped_env.vehicle
    ego.follow_road()
    return float(np.clip(ego.steering_control(ego.target_lane_index), -math.pi / 4.0, math.pi / 4.0))


def steering_bounds(
    unwrapped_env,
    lane_steering_margin: float | None = None,
) -> tuple[float, float]:
    steering_min = -math.pi / 4.0
    steering_max = math.pi / 4.0
    if lane_steering_margin is not None:
        lane_steering = lane_following_steering(unwrapped_env)
        steering_min = max(steering_min, lane_steering - lane_steering_margin)
        steering_max = min(steering_max, lane_steering + lane_steering_margin)
    return steering_min, steering_max


def lane_aware_control_bounds(
    unwrapped_env,
    dynamics,
    lane_steering_margin: float | None = None,
) -> list[tuple[float, float]]:
    bounds = control_bounds(dynamics)
    steering_min, steering_max = steering_bounds(unwrapped_env, lane_steering_margin)
    return [(steering_min, steering_max), bounds[1]]


def clip_control(
    unwrapped_env,
    control: np.ndarray,
    lane_steering_margin: float | None = None,
) -> np.ndarray:
    steering_min, steering_max = steering_bounds(unwrapped_env, lane_steering_margin)
    steering = float(np.clip(control[0], steering_min, steering_max))
    acceleration = float(np.clip(control[1], -5.0, 5.0))
    speed = float(unwrapped_env.vehicle.speed)
    if speed <= 1e-3:
        acceleration = max(acceleration, 0.0)
    if speed >= 32.0 - 1e-3:
        acceleration = min(acceleration, 0.0)
    return np.array([steering, acceleration], dtype=np.float64)


def hj_rate(context: SafetyContext, dynamics, control: np.ndarray) -> float:
    if context.state is None or context.dvds is None:
        return 0.0

    control_t = torch.tensor(control, dtype=torch.float32).view(1, 2)
    control_t = dynamics.clamp_control(context.state, control_t)
    disturbances = dynamics._control_vertices(context.state.device)
    scores = []
    for disturbance in disturbances:
        disturbance_t = disturbance.view(1, 2)
        dsdt = dynamics.dsdt(context.state, control_t, disturbance_t)
        scores.append(torch.sum(context.dvds * dsdt, dim=-1))
    return float(torch.min(torch.stack(scores, dim=-1), dim=-1).values.item())


def smooth_constraint(context: SafetyContext, dynamics, control: np.ndarray, gamma: float) -> float:
    return hj_rate(context, dynamics, control) + gamma * context.value


def hj_bang_bang_control(context: SafetyContext, dynamics) -> np.ndarray:
    if context.state is None or context.dvds is None:
        return np.zeros(2, dtype=np.float64)
    control = dynamics.optimal_control(context.state, context.dvds).detach().cpu().numpy()[0]
    return np.asarray(control, dtype=np.float64)


def hj_bang_bang_control_with_bounds(
    context: SafetyContext,
    dynamics,
    bounds: list[tuple[float, float]],
) -> np.ndarray:
    if context.state is None or context.dvds is None:
        return np.zeros(2, dtype=np.float64)

    controls = torch.tensor(
        [
            [bounds[0][0], bounds[1][0]],
            [bounds[0][0], bounds[1][1]],
            [bounds[0][1], bounds[1][0]],
            [bounds[0][1], bounds[1][1]],
        ],
        device=context.state.device,
        dtype=context.state.dtype,
    )
    disturbances = dynamics._control_vertices(context.state.device)

    control_scores = []
    for control in controls:
        control_batch = control.expand(*context.state.shape[:-1], 2)
        disturbance_scores = []
        for disturbance in disturbances:
            disturbance_batch = disturbance.expand(*context.state.shape[:-1], 2)
            dsdt = dynamics.dsdt(context.state, control_batch, disturbance_batch)
            disturbance_scores.append(torch.sum(context.dvds * dsdt, dim=-1))
        disturbance_scores = torch.stack(disturbance_scores, dim=-1)
        control_scores.append(torch.min(disturbance_scores, dim=-1).values)

    control_scores = torch.stack(control_scores, dim=-1)
    best_idx = torch.argmax(control_scores, dim=-1)
    return controls[best_idx].detach().cpu().numpy()[0].astype(np.float64)


def project_smooth_control(
    context: SafetyContext,
    dynamics,
    nominal_control: np.ndarray,
    gamma: float,
    bounds: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray | None, float | None]:
    if bounds is None:
        bounds = control_bounds(dynamics)
    nominal = np.array(nominal_control, dtype=np.float64)

    def objective(control):
        delta = np.asarray(control) - nominal
        return float(delta @ delta)

    def constraint(control):
        return smooth_constraint(context, dynamics, np.asarray(control), gamma)

    result = minimize(
        objective,
        np.clip(nominal, [b[0] for b in bounds], [b[1] for b in bounds]),
        method="SLSQP",
        bounds=bounds,
        constraints=[{"type": "ineq", "fun": constraint}],
        options={"maxiter": 50, "ftol": 1e-6, "disp": False},
    )
    if not result.success:
        return None, None

    control = np.asarray(result.x, dtype=np.float64)
    value = constraint(control)
    if value < -1e-5:
        return None, value
    return control, value


def choose_controller_action(
    env,
    policy,
    obs,
    context: SafetyContext,
    filter_name: str,
    gamma: float,
    value_threshold: float,
    lane_steering_margin: float,
    dynamics,
) -> ControllerDecision:
    nominal_action, _states = policy.predict(obs, deterministic=True)
    nominal_action = scalar_action(nominal_action)

    if context.dangerous_idx is None:
        return ControllerDecision(MODE_DQN, nominal_action, None, None, "nominal-no-traffic")

    if filter_name == FILTER_LEAST_RESTRICTIVE:
        if context.value >= value_threshold:
            return ControllerDecision(MODE_DQN, nominal_action, None, None, "nominal")
        bounds = lane_aware_control_bounds(env.unwrapped, dynamics, lane_steering_margin)
        control = clip_control(
            env.unwrapped,
            hj_bang_bang_control_with_bounds(context, dynamics, bounds),
            lane_steering_margin,
        )
        return ControllerDecision(
            MODE_HJ_BANG,
            None,
            control,
            smooth_constraint(context, dynamics, control, gamma=0.0),
            "hj-bang-bang",
        )

    if filter_name == FILTER_SMOOTH:
        nominal_control = nominal_continuous_control(env, nominal_action)
        nominal_constraint = smooth_constraint(context, dynamics, nominal_control, gamma)
        if nominal_constraint >= 0.0:
            return ControllerDecision(
                MODE_DQN,
                nominal_action,
                None,
                nominal_constraint,
                "nominal-satisfies-smooth",
            )

        projected_control, projected_constraint = project_smooth_control(
            context,
            dynamics,
            nominal_control,
            gamma,
            lane_aware_control_bounds(env.unwrapped, dynamics, lane_steering_margin),
        )
        if projected_control is not None:
            projected_control = clip_control(
                env.unwrapped,
                projected_control,
                lane_steering_margin,
            )
            return ControllerDecision(
                MODE_HJ_PROJECTED,
                None,
                projected_control,
                projected_constraint,
                "smooth-projection",
            )

        bounds = lane_aware_control_bounds(env.unwrapped, dynamics, lane_steering_margin)
        control = clip_control(
            env.unwrapped,
            hj_bang_bang_control_with_bounds(context, dynamics, bounds),
            lane_steering_margin,
        )
        return ControllerDecision(
            MODE_HJ_FALLBACK,
            None,
            control,
            smooth_constraint(context, dynamics, control, gamma),
            "hj-bang-bang-fallback",
        )

    raise ValueError(f"Unknown filter: {filter_name}")


def step_with_decision(env, decision: ControllerDecision):
    if decision.mode == MODE_DQN:
        return env.step(decision.nominal_action)
    if decision.control is None:
        raise ValueError(f"Continuous control missing for decision {decision.mode}")
    return step_with_continuous_ego_control(env, decision.control)


def format_decision_log(seed: int, step_count: int, filter_name: str, context: SafetyContext, decision: ControllerDecision) -> str:
    control = "nominal"
    if decision.control is not None:
        control = f"delta={decision.control[0]:+.3f}, a={decision.control[1]:+.3f}"
    constraint = "n/a" if decision.smooth_constraint is None else f"{decision.smooth_constraint:.3f}"
    return (
        f"  seed={seed:02d} step={step_count:02d}: filter={filter_name} "
        f"V={context.value:.3f} min={context.dangerous_idx} -> {decision.mode} "
        f"source={decision.source} constraint={constraint}; {control}"
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
    value_threshold: float,
    lane_steering_margin: float,
) -> dict:
    obs, _info = env.reset(seed=seed)

    rollout_frames: list[np.ndarray] = []
    frames_values: list[np.ndarray] = []
    ego_xy: list[tuple[float, float]] = []
    others_xy_per_frame: list[list[tuple[float, float]]] = []
    title_suffix_per_frame: list[str] = []
    controller_modes: list[str] = []
    controls: list[tuple[float, float] | None] = []

    terminated = False
    truncated = False
    crashed = False
    step_count = 0

    while not (terminated or truncated):
        states = src.extract_vehicle_states(env)
        values, context = evaluate_brt_frame(states, dynamics, vf_model, px_grid, py_grid)
        decision = choose_controller_action(
            env,
            policy,
            obs,
            context,
            filter_name,
            gamma,
            value_threshold,
            lane_steering_margin,
            dynamics,
        )

        ego = states[0]
        others = states[1:]
        rollout_frames.append(env.render())
        frames_values.append(values)
        ego_xy.append((ego.px, ego.py))
        others_xy_per_frame.append([(other.px, other.py) for other in others])
        title_suffix_per_frame.append(title_suffix(filter_name, decision, context))
        controller_modes.append(decision.mode)
        controls.append(None if decision.control is None else tuple(decision.control.tolist()))

        if decision.mode != MODE_DQN:
            print(format_decision_log(seed, step_count, filter_name, context, decision))

        obs, _reward, terminated, truncated, info = step_with_decision(env, decision)
        crashed = bool(info.get("crashed", crashed))
        step_count += 1

    terminal_states = src.extract_vehicle_states(env)
    terminal_decision = ControllerDecision("terminal", None, None, None, "terminal")
    capture_frame_data(
        env,
        terminal_states,
        dynamics,
        vf_model,
        px_grid,
        py_grid,
        filter_name,
        terminal_decision,
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
        "controls": controls,
        "crashed": crashed,
    }


def rollout_with_nominal_controller(env, policy, seed: int) -> dict:
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
        return f"hj_smooth_gamma_{gamma_label(gamma)}_seed_{seed:02d}"
    return f"hj_least_restrictive_seed_{seed:02d}"


def main() -> None:
    args = parse_args()
    if args.lane_steering_margin < 0.0:
        raise ValueError("--lane-steering-margin must be nonnegative.")
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
        f"filter={args.filter}; gamma={args.gamma:g}; "
        f"value_threshold={args.value_threshold:g}; "
        f"lane_steering_margin={args.lane_steering_margin:g}."
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
            bu.save_rollout_video(nominal_output_path, nominal_rollout["rollout_frames"])

            print(f"Running HJ safety rollout for seed={seed:02d} with filter={args.filter}...")
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
                args.value_threshold,
                args.lane_steering_margin,
            )
            modes = rollout["controller_modes"]
            dqn_steps = modes.count(MODE_DQN)
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
