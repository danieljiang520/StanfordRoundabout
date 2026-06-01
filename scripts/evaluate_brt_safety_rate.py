"""Compare nominal DQN-GNN safety rate against the BRT safety wrapper.

This script runs paired non-rendered rollouts: each seed is evaluated once with
the nominal DQN policy and once with the BRT-triggered stop override. Results
are written to a timestamped run directory under roundabout_dqn_gnn/safety_eval.
"""

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
import numpy as np
from stable_baselines3 import DQN

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

# import brt_utils as bu  # noqa: E402
import brt_utils_body_geometry as bu
import src  # registers VariableRoundabout-v0  # noqa: E402


MIN_TRAFFIC_VEHICLES = 1
MAX_TRAFFIC_VEHICLES = 7
SIMULATION_FREQUENCY = 15
POLICY_FREQUENCY = 5
SAFETY_VALUE_THRESHOLD = 0
DEFAULT_NUM_ROLLOUTS = 500

OUTPUT_ROOT = bu.REPO_ROOT / "roundabout_dqn_gnn" / "safety_eval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate nominal DQN vs BRT safety-controller crash rates."
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
    return parser.parse_args()


def make_output_dir(timestamp: datetime, traffic_vehicles: int) -> Path:
    base_name = f"{timestamp.strftime('%m%d%y_%H%M')}_traffic_{traffic_vehicles:02d}"
    output_dir = OUTPUT_ROOT / base_name
    suffix = 1
    while output_dir.exists():
        output_dir = OUTPUT_ROOT / f"{base_name}_{suffix:02d}"
        suffix += 1
    output_dir.mkdir(parents=True)
    return output_dir


def min_brt_value_at_ego(states, dynamics, vf_model) -> float:
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


def rollout_with_safety_controller(env, policy, dynamics, vf_model, seed: int) -> dict[str, Any]:
    obs, _info = env.reset(seed=seed)
    idle_action = env.unwrapped.action_type.actions_indexes["IDLE"]

    saved_controller_state: tuple[float, int] | None = None
    terminated = False
    truncated = False
    crashed = False
    step_count = 0
    stop_steps = 0
    total_reward = 0.0
    min_value_seen = math.inf

    while not (terminated or truncated):
        ego_value = min_brt_value_at_ego(src.extract_vehicle_states(env), dynamics, vf_model)
        min_value_seen = min(min_value_seen, ego_value)
        mode = "STOP" if ego_value < SAFETY_VALUE_THRESHOLD else "DQN"

        controlled_ego = env.unwrapped.vehicle
        if mode == "STOP":
            stop_steps += 1
            if saved_controller_state is None:
                saved_controller_state = (
                    float(controlled_ego.target_speed),
                    int(controlled_ego.speed_index),
                )
            controlled_ego.speed = 0.0
            controlled_ego.target_speed = 0.0
            action = idle_action
        else:
            if saved_controller_state is not None:
                controlled_ego.target_speed, controlled_ego.speed_index = saved_controller_state
                saved_controller_state = None
            action, _states = policy.predict(obs, deterministic=True)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        crashed = bool(info.get("crashed", crashed))
        step_count += 1

    return {
        "crashed": crashed,
        "steps": step_count,
        "stop_steps": stop_steps,
        "stop_fraction": stop_steps / step_count if step_count else 0.0,
        "min_value": min_value_seen if math.isfinite(min_value_seen) else None,
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

    return {
        "experiment": "brt_safety_controller_vs_nominal_dqn",
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
        "controller_config": {
            "traffic_vehicles": traffic_vehicles,
            "simulation_frequency": SIMULATION_FREQUENCY,
            "policy_frequency": POLICY_FREQUENCY,
            "safety_value_threshold": SAFETY_VALUE_THRESHOLD,
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
        "brt_stop_steps",
        "brt_stop_fraction",
        "brt_min_value",
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
        f"{'Controller':<16} {'Safe':>6} {'Crashed':>8} "
        f"{'Safety Rate':>12} {'Crash Rate':>11} {'95% CI Crash Rate':>22}"
    )
    print("-" * 82)
    print(
        f"{'Nominal DQN':<16} {n - summary['nominal_crashes']:>6} "
        f"{summary['nominal_crashes']:>8} {pct(summary['nominal_safety_rate']):>12} "
        f"{pct(summary['nominal_crash_rate']):>11} "
        f"{ci(summary['nominal_crash_rate_ci95']):>22}"
    )
    print(
        f"{'BRT Safety':<16} {n - summary['brt_crashes']:>6} "
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
    print(f"Results saved to: {summary['output_dir']}")


def evaluate_traffic_count(
    *,
    traffic_vehicles: int,
    num_rollouts: int,
    start_seed: int,
    timestamp: datetime,
    dynamics,
    vf_model,
    policy,
) -> dict[str, Any]:
    output_dir = make_output_dir(timestamp, traffic_vehicles)
    csv_path = output_dir / f"rollouts_{num_rollouts}.csv"
    summary_path = output_dir / "summary.json"

    print()
    print(f"Evaluating {num_rollouts} paired rollouts with {traffic_vehicles} traffic vehicles...")
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
            print(" BRT...", end="", flush=True)
            brt = rollout_with_safety_controller(env, policy, dynamics, vf_model, seed)
            print(
                f" nominal_crashed={nominal['crashed']} "
                f"brt_crashed={brt['crashed']} "
                f"stop_steps={brt['stop_steps']}"
            )

            rows.append(
                {
                    "seed": seed,
                    "nominal_crashed": bool(nominal["crashed"]),
                    "brt_crashed": bool(brt["crashed"]),
                    "nominal_steps": int(nominal["steps"]),
                    "brt_steps": int(brt["steps"]),
                    "brt_stop_steps": int(brt["stop_steps"]),
                    "brt_stop_fraction": float(brt["stop_fraction"]),
                    "brt_min_value": brt["min_value"],
                    "nominal_total_reward": float(nominal["total_reward"]),
                    "brt_total_reward": float(brt["total_reward"]),
                }
            )
    finally:
        env.close()

    write_csv(csv_path, rows)
    summary = summarize(rows, timestamp, output_dir, traffic_vehicles)
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
                dynamics=dynamics,
                vf_model=vf_model,
                policy=policy,
            )
        )

    print()
    print("Traffic sweep complete.")
    print(
        f"{'Traffic':>7} {'Nominal Crash':>15} {'BRT Crash':>11} "
        f"{'Safety Improvement':>20} {'Output Folder'}"
    )
    print("-" * 92)
    for summary in summaries:
        traffic_vehicles = summary["controller_config"]["traffic_vehicles"]
        print(
            f"{traffic_vehicles:>7} "
            f"{100.0 * summary['nominal_crash_rate']:>14.2f}% "
            f"{100.0 * summary['brt_crash_rate']:>10.2f}% "
            f"{100.0 * summary['absolute_safety_rate_improvement']:>19.2f}% "
            f"{summary['output_dir']}"
        )


if __name__ == "__main__":
    main()
