"""Visualize the TwoCar8D BRT over a 2-car roundabout rollout.

Runs one episode of the DQN-GNN policy in VariableRoundabout-v0 with one
traffic vehicle, then animates a (px1, py1) slice of the trained BRT —
re-sliced each frame at the live (psi1, v1, px2, py2, psi2, v2) — with a
green dot tracking the ego and a blue dot tracking the other vehicle.

Shared helpers live in scripts/brt_utils.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from tqdm import tqdm

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(SCRIPTS_DIR.parent))

import brt_utils as bu  # noqa: E402
import src  # registers VariableRoundabout-v0  # noqa: E402


TRAFFIC_VEHICLES = 1
SEED_CANDIDATES = list(range(20))

BRT_OUTPUT_PATH = bu.VIDEOS_DIR / "brt_overlay.mp4"
ROLLOUT_OUTPUT_PATH = bu.VIDEOS_DIR / "rollout.mp4"
SIDEBYSIDE_OUTPUT_PATH = bu.VIDEOS_DIR / "rollout_and_brt.mp4"


def main():
    print("Loading DeepReach BRT...")
    dynamics = bu.build_cpu_dynamics()
    vf_model = bu.load_value_network(dynamics)

    config = bu.make_roundabout_config(TRAFFIC_VEHICLES)

    print(f"Selecting rollout seed across {len(SEED_CANDIDATES)} candidates...")
    selection_env = gym.make("VariableRoundabout-v0", render_mode=None, config=config)
    policy = DQN.load(str(bu.MODEL_PATH), env=selection_env)
    best_seed = None
    best_n = -1
    for seed in SEED_CANDIDATES:
        candidate = src.rollout_with_states(selection_env, policy, deterministic=True, seed=seed)
        n = len(candidate["states"])
        if n > best_n:
            best_n = n
            best_seed = seed
    selection_env.close()
    print(f"  picked seed={best_seed} with {best_n} frames")

    print(f"Re-running seed {best_seed} with rendering to capture rollout frames...")
    render_env = gym.make("VariableRoundabout-v0", render_mode="rgb_array", config=config)
    rollout = src.rollout_with_states(
        render_env, policy, deterministic=True, seed=best_seed, capture_frames=True
    )
    render_env.close()
    n_frames = len(rollout["states"])
    rollout_frames = rollout["frames"]
    print(f"  captured {n_frames} states, {len(rollout_frames)} rendered frames, crashed={rollout['crashed']}")

    px_axis, py_axis, px_grid, py_grid = bu.make_grid()

    print("Querying BRT along trajectory...")
    frames_values = []
    ego_xy = []
    others_xy_per_frame = []
    for states in tqdm(rollout["states"]):
        ego = states[0]
        other = states[1] if len(states) > 1 else states[0]
        values = bu.query_slice_single(
            dynamics, vf_model, px_grid, py_grid,
            ego_psi=ego.psi, ego_v=ego.v, other=other.as_array(),
        )
        frames_values.append(values)
        ego_xy.append((ego.px, ego.py))
        others_xy_per_frame.append([(other.px, other.py)])

    raw_vbar = float(np.max(np.abs(np.concatenate([v.flatten() for v in frames_values]))))
    print(f"  raw value magnitude range: ±{raw_vbar:.2f}; clipping colormap to ±{bu.VALUE_CMAP_LIMIT}")

    bu.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    bu.save_brt_video(BRT_OUTPUT_PATH, px_axis, py_axis, frames_values,
                      ego_xy, others_xy_per_frame, n_frames)
    bu.save_rollout_video(ROLLOUT_OUTPUT_PATH, rollout_frames)
    bu.save_sidebyside_video(SIDEBYSIDE_OUTPUT_PATH, rollout_frames,
                             px_axis, py_axis, frames_values,
                             ego_xy, others_xy_per_frame, n_frames)
    print("Done.")


if __name__ == "__main__":
    main()
