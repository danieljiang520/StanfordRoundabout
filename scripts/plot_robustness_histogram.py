"""
Collect trajectory metrics across envs × trajs (nominal + collision),
compute robustness per trajectory (collisions => 0), and plot histogram.
Use --optimize to fit weights on nominals so their distribution is exponential.
Run from project root: python scripts/plot_robustness_histogram.py [--optimize]
"""
import argparse
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

import highway_env  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.robustness import (
    trajectory_metrics_from_rollout,
    compute_robustness,
    optimize_weights_for_exponential,
)

NUM_ENVIRONMENTS = 30
NUM_TRAJECTORIES = 5
MODEL_PATH = "roundabout_dqn/model"

# Default: uniform over the five components
DEFAULT_WEIGHTS = {
    "safety": 0.2,
    "stability": 0.2,
    "efficiency": 0.2,
    "road": 0.2,
    "hard_brakes": 0.2,
}


def get_action(env, obs):
    action, _ = model.predict(obs, deterministic=False)
    return action


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot robustness histogram (optionally optimize weights).")
    parser.add_argument("--optimize", action="store_true", help="Optimize weights so distribution is exponential (differentiate safe vs less safe)")
    parser.add_argument("--lam", type=float, default=2.5, help="Exponential rate for target (default 2.5; higher = more mass at high robustness)")
    parser.add_argument(
        "--trajectories",
        choices=("nominal", "all"),
        default="all",
        help="Plot nominal (no collision) only, or all trajectories (default: all)",
    )
    parser.add_argument(
        "--min-weight",
        type=float,
        default=0.05,
        metavar="W",
        help="Minimum weight per component when optimizing (default 0.05); use 0 to allow near-zero weights",
    )
    args = parser.parse_args()

    env = gym.make("roundabout-v0", render_mode="rgb_array")
    model = DQN.load(MODEL_PATH, env=env)

    all_metrics = []
    for env_idx in range(NUM_ENVIRONMENTS):
        seed = env_idx * 100
        for _ in range(NUM_TRAJECTORIES):
            metrics = trajectory_metrics_from_rollout(env, get_action, seed=seed)
            all_metrics.append(metrics)

    nominal_metrics = [m for m in all_metrics if m["success"] == 1.0]
    collision_metrics = [m for m in all_metrics if m["success"] != 1.0]
    n_nominal, n_collision = len(nominal_metrics), len(collision_metrics)

    if args.optimize and not nominal_metrics:
        print("No nominal (success) trajectories; cannot optimize weights.")
        env.close()
        sys.exit(1)

    if args.optimize:
        weights, loss, nominal_scores = optimize_weights_for_exponential(
            nominal_metrics, lam=args.lam, min_weight=args.min_weight
        )
        nominal_scores = list(nominal_scores)
        print("Optimized weights (exponential-shaped distribution):")
        for k, v in weights.items():
            print(f"  {k}: {v:.4f}")
        print(f"  Histogram MSE to target (nominals): {loss:.6f}")
        print(f"  Nominal robustness: mean={np.mean(nominal_scores):.4f}, std={np.std(nominal_scores):.4f}, range=[{np.min(nominal_scores):.4f}, {np.max(nominal_scores):.4f}]")
    else:
        weights = DEFAULT_WEIGHTS

    # Which trajectories to include in histogram
    plot_metrics = all_metrics if args.trajectories == "all" else nominal_metrics
    if not plot_metrics:
        print(f"No trajectories to plot (--trajectories={args.trajectories}).")
        env.close()
        sys.exit(1)
    scores = [compute_robustness(m, weights=weights) for m in plot_metrics]

    if args.trajectories == "all":
        title = f"All trajectories (n={len(all_metrics)}): nominal {n_nominal}, collision {n_collision}"
    else:
        title = f"Nominal only (n={len(nominal_metrics)})"
    title += ", exponential-fit weights" if args.optimize else ", uniform weights"

    fig, ax = plt.subplots()
    ax.hist(scores, bins=min(30, max(10, len(scores) // 5)), density=True, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Robustness")
    ax.set_ylabel("Frequency (density)")
    ax.set_title(title)
    ax.set_xlim(left=0)
    fig.tight_layout()
    out_dir = Path("roundabout_dqn")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "robustness_histogram.png", dpi=150)
    plt.show()

    print(f"Nominal: {n_nominal}, Collision: {n_collision}, Total: {len(all_metrics)}")
    print(f"Plotted: {args.trajectories} (n={len(plot_metrics)})")
    print(f"Robustness: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
    env.close()