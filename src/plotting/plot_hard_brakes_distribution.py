"""
Plot distribution of number of hard brakes for nominal (success) trajectories.
Run from project root: python scripts/plot_hard_brakes_distribution.py
"""
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN

import highway_env  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.robustness import trajectory_metrics_from_rollout

NUM_ENVIRONMENTS = 30
NUM_TRAJECTORIES = 5
MODEL_PATH = "roundabout_dqn/model"


def get_action(env, obs):
    action, _ = model.predict(obs, deterministic=False)
    return action


if __name__ == "__main__":
    env = gym.make("roundabout-v0", render_mode="rgb_array")
    model = DQN.load(MODEL_PATH, env=env)

    all_metrics = []
    for env_idx in range(NUM_ENVIRONMENTS):
        seed = env_idx * 100
        for _ in range(NUM_TRAJECTORIES):
            metrics = trajectory_metrics_from_rollout(env, get_action, seed=seed)
            all_metrics.append(metrics)

    nominal_metrics = [m for m in all_metrics if m["success"] == 1.0]
    hard_brakes_counts = [int(m["hard_brakes"]) for m in all_metrics]

    env.close()

    if not hard_brakes_counts:
        print("No trajectories; nothing to plot.")
        sys.exit(0)

    fig, ax = plt.subplots()
    bins = np.arange(-0.5, max(hard_brakes_counts) + 1.5, 1)
    ax.hist(hard_brakes_counts, bins=bins, edgecolor="black", alpha=0.7, density=False)
    ax.set_xlabel("Number of hard brakes")
    ax.set_ylabel("Count")
    n_nominal = len(nominal_metrics)
    n_collision = len(all_metrics) - n_nominal
    ax.set_title(f"Hard brakes per episode (n={len(all_metrics)}: nominal {n_nominal}, collision {n_collision})")
    ax.set_xticks(np.unique(hard_brakes_counts).astype(int))
    fig.tight_layout()

    out_dir = Path("roundabout_dqn")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "hard_brakes_distribution.png", dpi=150)
    plt.show()

    print(f"Total trajectories: {len(all_metrics)} (nominal {n_nominal}, collision {n_collision})")
    print(f"Hard brakes: mean={np.mean(hard_brakes_counts):.2f}, std={np.std(hard_brakes_counts):.2f}, min={min(hard_brakes_counts)}, max={max(hard_brakes_counts)}")
