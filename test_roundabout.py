"""
Main entry point: run roundabout policy rollouts and report robustness score per trajectory.
"""
import gymnasium as gym
from stable_baselines3 import DQN

import highway_env  # noqa: F401

from src.robustness import (
    trajectory_metrics_from_rollout,
    compute_robustness,
)

NUM_TRAJECTORIES = 30
MODEL_PATH = "roundabout_dqn/model"


def get_action(env, obs):
    action, _ = model.predict(obs, deterministic=True)
    return action


if __name__ == "__main__":
    env = gym.make("roundabout-v0", render_mode="rgb_array")
    model = DQN.load(MODEL_PATH, env=env)

    scores = []
    for i in range(NUM_TRAJECTORIES):
        seed = i * 100
        metrics = trajectory_metrics_from_rollout(env, get_action, seed=seed)
        score = compute_robustness(metrics)
        scores.append(score)
        print(f"  traj {i:3d}  seed {seed:5d}  robustness {score:.4f}  success {metrics['success']:.0f}  min_dist {metrics['min_distance']:.3f}")

    print(f"\n  mean robustness: {sum(scores) / len(scores):.4f}  (n={len(scores)})")
    env.close()
