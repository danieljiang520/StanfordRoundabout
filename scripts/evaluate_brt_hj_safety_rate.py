"""Evaluate nominal DQN-GNN safety rate against HJ optimal BRT safety control."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import DQN

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import brt_utils_body_geometry as bu  # noqa: E402
import src  # registers VariableRoundabout-v0  # noqa: E402
from visualize_brt_hj_safety_rollouts_multicar import (  # noqa: E402
    DEFAULT_GAMMA,
    DEFAULT_LANE_STEERING_MARGIN,
    DEFAULT_VALUE_THRESHOLD,
    FILTER_LEAST_RESTRICTIVE,
    FILTER_SMOOTH,
    MODE_DQN,
    MODE_HJ_BANG,
    MODE_HJ_FALLBACK,
    MODE_HJ_PROJECTED,
    POLICY_FREQUENCY,
    SIMULATION_FREQUENCY,
    choose_controller_action,
    step_with_decision,
    value_and_gradient_at_ego,
)


MIN_TRAFFIC_VEHICLES = 1
MAX_TRAFFIC_VEHICLES = 7
DEFAULT_NUM_ROLLOUTS = 100
DEFAULT_FILTER = FILTER_SMOOTH
OUTPUT_ROOT = bu.REPO_ROOT / "roundabout_dqn_gnn" / "safety_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate nominal DQN vs HJ optimal BRT safety-controller crash rates."
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=DEFAULT_NUM_ROLLOUTS,
        help=f"Number of paired seeds to evaluate. Default: {DEFAULT_NUM_ROLLOUTS}.",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
        help="First seed in the paired rollout sequence. Default: 0.",
    )
    parser.add_argument(
        "--min-traffic-vehicles",
        type=int,
        default=MIN_TRAFFIC_VEHICLES,
        help=f"Smallest traffic vehicle count to evaluate. Default: {MIN_TRAFFIC_VEHICLES}.",
    )
    parser.add_argument(
        "--max-traffic-vehicles",
        type=int,
        default=MAX_TRAFFIC_VEHICLES,
        help=f"Largest traffic vehicle count to evaluate. Default: {MAX_TRAFFIC_VEHICLES}.",
    )
    parser.add_argument(
        "--filter",
        choices=[FILTER_LEAST_RESTRICTIVE, FILTER_SMOOTH],
        default=DEFAULT_FILTER,
        help=f"Safety filter to evaluate. Default: {DEFAULT_FILTER}.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help=f"Smooth blending gamma. Used by --filter smooth. Default: {DEFAULT_GAMMA:g}.",
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
    return parser.parse_args()


def label_float(value: float) -> str:
    return f"{value:g}".replace("-", "neg").replace(".", "p")


def make_output_dir(
    timestamp: datetime,
    traffic_vehicles: int,
    filter_name: str,
    gamma: float,
    value_threshold: float,
) -> Path:
    if filter_name == FILTER_SMOOTH:
        suffix = f"hj_smooth_gamma_{label_float(gamma)}"
    else:
        suffix = f"hj_least_restrictive_threshold_{label_float(value_threshold)}"
    base_name = f"{timestamp.strftime('%m%d%y_%H%M')}_traffic_{traffic_vehicles:02d}_{suffix}"
    output_dir = OUTPUT_ROOT / base_name
    suffix_idx = 1
    while output_dir.exists():
        output_dir = OUTPUT_ROOT / f"{base_name}_{suffix_idx:02d}"
        suffix_idx += 1
    output_dir.mkdir(parents=True)
    return output_dir


def rollout_with_nominal_controller(env, policy, seed: int) -> dict[str, Any]:
    obs, _info = env.reset(seed=seed)
    terminated = False
    truncated = False
    crashed = False
    step_count = 0
    total_reward = 0.0

    while not (terminated or truncated):
        action, _states = policy.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        crashed = bool(info.get("crashed", crashed))
        step_count += 1

    return {
        "crashed": crashed,
        "steps": step_count,
        "total_reward": total_reward,
    }


def empty_mode_counts() -> dict[str, int]:
    return {
        MODE_HJ_BANG: 0,
        MODE_HJ_PROJECTED: 0,
        MODE_HJ_FALLBACK: 0,
    }


def rollout_with_safety_controller(
    env,
    policy,
    dynamics,
    vf_model,
    seed: int,
    filter_name: str,
    gamma: float,
    value_threshold: float,
    lane_steering_margin: float,
) -> dict[str, Any]:
    obs, _info = env.reset(seed=seed)

    mode_counts = empty_mode_counts()
    terminated = False
    truncated = False
    crashed = False
    step_count = 0
    safe_steps = 0
    total_reward = 0.0
    min_value_seen = math.inf
    min_constraint_seen = math.inf

    while not (terminated or truncated):
        states = src.extract_vehicle_states(env)
        context = value_and_gradient_at_ego(states, dynamics, vf_model)
        min_value_seen = min(min_value_seen, context.value)
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

        if decision.smooth_constraint is not None:
            min_constraint_seen = min(min_constraint_seen, decision.smooth_constraint)
        if decision.mode != MODE_DQN:
            safe_steps += 1
            mode_counts[decision.mode] = mode_counts.get(decision.mode, 0) + 1

        obs, reward, terminated, truncated, info = step_with_decision(env, decision)
        total_reward += float(reward)
        crashed = bool(info.get("crashed", crashed))
        step_count += 1

    return {
        "crashed": crashed,
        "steps": step_count,
        "safe_steps": safe_steps,
        "safe_fraction": safe_steps / step_count if step_count else 0.0,
        "mode_counts": mode_counts,
        "min_value": min_value_seen if math.isfinite(min_value_seen) else None,
        "min_constraint": min_constraint_seen if math.isfinite(min_constraint_seen) else None,
        "total_reward": total_reward,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p_hat = successes / total
    denom = 1.0 + z * z / total
    center = (p_hat + z * z / (2.0 * total)) / denom
    half_width = (
        z
        * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z * z / (4.0 * total * total)))
        / denom
    )
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def exact_mcnemar_p_value(n01: int, n10: int) -> float | None:
    discordant = n01 + n10
    if discordant == 0:
        return None
    tail = sum(math.comb(discordant, k) for k in range(0, min(n01, n10) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def summarize(
    rows: list[dict[str, Any]],
    timestamp: datetime,
    output_dir: Path,
    traffic_vehicles: int,
    filter_name: str,
    gamma: float,
    value_threshold: float,
    lane_steering_margin: float,
) -> dict[str, Any]:
    n = len(rows)
    nominal_crashes = sum(bool(row["nominal_crashed"]) for row in rows)
    brt_crashes = sum(bool(row["brt_crashed"]) for row in rows)

    nominal_safe_brt_crash = sum(
        (not bool(row["nominal_crashed"])) and bool(row["brt_crashed"]) for row in rows
    )
    nominal_crash_brt_safe = sum(
        bool(row["nominal_crashed"]) and (not bool(row["brt_crashed"])) for row in rows
    )
    both_safe = sum(
        (not bool(row["nominal_crashed"])) and (not bool(row["brt_crashed"])) for row in rows
    )
    both_crashed = sum(
        bool(row["nominal_crashed"]) and bool(row["brt_crashed"]) for row in rows
    )

    nominal_crash_rate = nominal_crashes / n if n else 0.0
    brt_crash_rate = brt_crashes / n if n else 0.0
    nominal_safety_rate = 1.0 - nominal_crash_rate
    brt_safety_rate = 1.0 - brt_crash_rate
    mode_totals = {
        MODE_HJ_BANG: sum(int(row["brt_hj_bang_steps"]) for row in rows),
        MODE_HJ_PROJECTED: sum(int(row["brt_hj_projected_steps"]) for row in rows),
        MODE_HJ_FALLBACK: sum(int(row["brt_hj_fallback_steps"]) for row in rows),
    }

    return {
        "experiment": "brt_hj_safety_controller_vs_nominal_dqn",
        "run_timestamp": timestamp.isoformat(timespec="seconds"),
        "output_dir": str(output_dir),
        "num_rollouts": n,
        "seeds": [row["seed"] for row in rows],
        "nominal_crashes": nominal_crashes,
        "brt_crashes": brt_crashes,
        "nominal_crash_rate": nominal_crash_rate,
        "brt_crash_rate": brt_crash_rate,
        "nominal_safety_rate": nominal_safety_rate,
        "brt_safety_rate": brt_safety_rate,
        "absolute_safety_rate_improvement": brt_safety_rate - nominal_safety_rate,
        "nominal_crash_rate_ci95": wilson_interval(nominal_crashes, n),
        "brt_crash_rate_ci95": wilson_interval(brt_crashes, n),
        "paired_comparison_counts": {
            "both_safe": both_safe,
            "both_crashed": both_crashed,
            "nominal_safe_brt_crashed": nominal_safe_brt_crash,
            "nominal_crashed_brt_safe": nominal_crash_brt_safe,
        },
        "paired_exact_test": {
            "test": "exact_mcnemar_binomial",
            "p_value": exact_mcnemar_p_value(nominal_safe_brt_crash, nominal_crash_brt_safe),
        },
        "controller_mode_totals": mode_totals,
        "controller_config": {
            "traffic_vehicles": traffic_vehicles,
            "simulation_frequency": SIMULATION_FREQUENCY,
            "policy_frequency": POLICY_FREQUENCY,
            "filter": filter_name,
            "smooth_gamma": gamma,
            "value_threshold": value_threshold,
            "lane_steering_margin": lane_steering_margin,
            "control_bounds": {
                "steering": [-math.pi / 4.0, math.pi / 4.0],
                "acceleration": [-5.0, 5.0],
            },
            "nominal_policy": str(bu.MODEL_PATH),
            "brt_checkpoint": str(bu.CKPT_PATH),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "nominal_crashed",
        "brt_crashed",
        "nominal_steps",
        "brt_steps",
        "brt_safe_steps",
        "brt_safe_fraction",
        "brt_hj_bang_steps",
        "brt_hj_projected_steps",
        "brt_hj_fallback_steps",
        "brt_min_value",
        "brt_min_constraint",
        "nominal_total_reward",
        "brt_total_reward",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: dict[str, Any]) -> None:
    n = summary["num_rollouts"]
    traffic_vehicles = summary["controller_config"]["traffic_vehicles"]

    def pct(value: float) -> str:
        return f"{100.0 * value:6.2f}%"

    def ci(values: list[float]) -> str:
        return f"[{100.0 * values[0]:.2f}%, {100.0 * values[1]:.2f}%]"

    print()
    print(f"Evaluated {n} paired rollouts with {traffic_vehicles} traffic vehicles.")
    print()
    print(
        f"{'Controller':<18} {'Safe':>6} {'Crashed':>8} "
        f"{'Safety Rate':>12} {'Crash Rate':>11} {'95% CI Crash Rate':>22}"
    )
    print("-" * 84)
    print(
        f"{'Nominal DQN':<18} {n - summary['nominal_crashes']:>6} "
        f"{summary['nominal_crashes']:>8} {pct(summary['nominal_safety_rate']):>12} "
        f"{pct(summary['nominal_crash_rate']):>11} "
        f"{ci(summary['nominal_crash_rate_ci95']):>22}"
    )
    print(
        f"{'HJ BRT Safety':<18} {n - summary['brt_crashes']:>6} "
        f"{summary['brt_crashes']:>8} {pct(summary['brt_safety_rate']):>12} "
        f"{pct(summary['brt_crash_rate']):>11} "
        f"{ci(summary['brt_crash_rate_ci95']):>22}"
    )
    print()
    print(
        "Absolute safety-rate improvement: "
        f"{100.0 * summary['absolute_safety_rate_improvement']:+.2f} percentage points"
    )
    p_value = summary["paired_exact_test"]["p_value"]
    p_text = "undefined (no discordant pairs)" if p_value is None else f"{p_value:.6g}"
    print(f"Paired exact McNemar/binomial p-value: {p_text}")
    print(f"HJ mode totals: {summary['controller_mode_totals']}")
    print(f"Results saved to: {summary['output_dir']}")


def evaluate_traffic_count(
    *,
    traffic_vehicles: int,
    num_rollouts: int,
    start_seed: int,
    timestamp: datetime,
    filter_name: str,
    gamma: float,
    value_threshold: float,
    lane_steering_margin: float,
    dynamics,
    vf_model,
    policy,
) -> dict[str, Any]:
    output_dir = make_output_dir(timestamp, traffic_vehicles, filter_name, gamma, value_threshold)
    csv_path = output_dir / f"rollouts_{num_rollouts}.csv"
    summary_path = output_dir / "summary.json"

    print()
    print(
        f"Evaluating {num_rollouts} paired rollouts with {traffic_vehicles} traffic vehicles, "
        f"filter={filter_name}, gamma={gamma:g}, value_threshold={value_threshold:g}, "
        f"lane_steering_margin={lane_steering_margin:g}..."
    )
    config = bu.make_roundabout_config(traffic_vehicles)
    config["simulation_frequency"] = SIMULATION_FREQUENCY
    config["policy_frequency"] = POLICY_FREQUENCY
    env = gym.make("VariableRoundabout-v0", render_mode=None, config=config)

    rows: list[dict[str, Any]] = []
    try:
        seeds = range(start_seed, start_seed + num_rollouts)
        for rollout_index, seed in enumerate(seeds, start=1):
            print(f"[{rollout_index:>4}/{num_rollouts}] seed={seed}: nominal...", end="", flush=True)
            nominal = rollout_with_nominal_controller(env, policy, seed)
            print(" HJ BRT...", end="", flush=True)
            brt = rollout_with_safety_controller(
                env,
                policy,
                dynamics,
                vf_model,
                seed,
                filter_name,
                gamma,
                value_threshold,
                lane_steering_margin,
            )
            mode_counts = brt["mode_counts"]
            print(
                f" nominal_crashed={nominal['crashed']} "
                f"brt_crashed={brt['crashed']} "
                f"safe_steps={brt['safe_steps']} "
                f"modes={mode_counts}"
            )

            rows.append(
                {
                    "seed": seed,
                    "nominal_crashed": bool(nominal["crashed"]),
                    "brt_crashed": bool(brt["crashed"]),
                    "nominal_steps": int(nominal["steps"]),
                    "brt_steps": int(brt["steps"]),
                    "brt_safe_steps": int(brt["safe_steps"]),
                    "brt_safe_fraction": float(brt["safe_fraction"]),
                    "brt_hj_bang_steps": int(mode_counts[MODE_HJ_BANG]),
                    "brt_hj_projected_steps": int(mode_counts[MODE_HJ_PROJECTED]),
                    "brt_hj_fallback_steps": int(mode_counts[MODE_HJ_FALLBACK]),
                    "brt_min_value": brt["min_value"],
                    "brt_min_constraint": brt["min_constraint"],
                    "nominal_total_reward": float(nominal["total_reward"]),
                    "brt_total_reward": float(brt["total_reward"]),
                }
            )
    finally:
        env.close()

    write_csv(csv_path, rows)
    summary = summarize(
        rows,
        timestamp,
        output_dir,
        traffic_vehicles,
        filter_name,
        gamma,
        value_threshold,
        lane_steering_margin,
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print_summary(summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.num_rollouts <= 0:
        raise ValueError("--num-rollouts must be positive.")
    if args.min_traffic_vehicles <= 0:
        raise ValueError("--min-traffic-vehicles must be positive.")
    if args.max_traffic_vehicles < args.min_traffic_vehicles:
        raise ValueError("--max-traffic-vehicles must be >= --min-traffic-vehicles.")
    if args.gamma < 0.0:
        raise ValueError("--gamma must be nonnegative.")
    if args.lane_steering_margin < 0.0:
        raise ValueError("--lane-steering-margin must be nonnegative.")

    timestamp = datetime.now().astimezone()

    print("Loading DeepReach BRT and DQN-GNN policy...")
    dynamics = bu.build_cpu_dynamics()
    vf_model = bu.load_value_network(dynamics)
    policy = DQN.load(str(bu.MODEL_PATH))

    summaries = []
    for traffic_vehicles in range(args.min_traffic_vehicles, args.max_traffic_vehicles + 1):
        summaries.append(
            evaluate_traffic_count(
                traffic_vehicles=traffic_vehicles,
                num_rollouts=args.num_rollouts,
                start_seed=args.start_seed,
                timestamp=timestamp,
                filter_name=args.filter,
                gamma=args.gamma,
                value_threshold=args.value_threshold,
                lane_steering_margin=args.lane_steering_margin,
                dynamics=dynamics,
                vf_model=vf_model,
                policy=policy,
            )
        )

    print()
    print("Traffic sweep complete.")
    print(
        f"{'Traffic':>7} {'Nominal Crash':>15} {'HJ BRT Crash':>14} "
        f"{'Safety Improvement':>20} {'Output Folder'}"
    )
    print("-" * 102)
    for summary in summaries:
        traffic_vehicles = summary["controller_config"]["traffic_vehicles"]
        print(
            f"{traffic_vehicles:>7} "
            f"{100.0 * summary['nominal_crash_rate']:>14.2f}% "
            f"{100.0 * summary['brt_crash_rate']:>13.2f}% "
            f"{100.0 * summary['absolute_safety_rate_improvement']:>19.2f}% "
            f"{summary['output_dir']}"
        )


if __name__ == "__main__":
    main()
