"""Visualize the TwoCar8D BRT over a roundabout rollout with multiple traffic vehicles.

The TwoCar8D BRT is trained for a single other car. To apply it with N other
vehicles, we follow the most-threatening-obstacle pattern from
homework/AA276_GCP/hw3/problem3.py (Vs.argmin selecting one obstacle): each
frame, evaluate V at the ego's current state once per other vehicle, pick
the vehicle with the lowest V (the most dangerous one), then plot that
single vehicle's (px1, py1) slice and only its blue dot.
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


TRAFFIC_VEHICLES = 3
SEED_CANDIDATES = list(range(20))

SIDEBYSIDE_OUTPUT_PATH = bu.VIDEOS_DIR / "rollout_and_brt_multicar.mp4"


def main():
    print("Loading DeepReach BRT...")
    dynamics = bu.build_cpu_dynamics()
    vf_model = bu.load_value_network(dynamics)

    config = bu.make_roundabout_config(TRAFFIC_VEHICLES)

    print(f"Selecting rollout seed across {len(SEED_CANDIDATES)} candidates "
          f"with TRAFFIC_VEHICLES={TRAFFIC_VEHICLES}...")
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

    print("Querying BRT along trajectory (most-dangerous other per frame)...")
    frames_values = []
    ego_xy = []
    others_xy_per_frame = []
    title_suffix_per_frame = []
    for states in tqdm(rollout["states"]):
        ego = states[0]
        others = states[1:]
        ego_xy.append((ego.px, ego.py))
        others_xy_per_frame.append([(o.px, o.py) for o in others])

        if not others:
            values = np.full(px_grid.shape, bu.VALUE_CMAP_LIMIT, dtype=np.float32)
            title_suffix_per_frame.append("no other vehicles")
            frames_values.append(values)
            continue

        others_arr = np.stack([o.as_array() for o in others])  # (K, 4)
        # Pick the most dangerous other vehicle = lowest V at the ego's actual state.
        ego_arr = ego.as_array()
        v_at_ego = bu.values_at_ego_state(dynamics, vf_model, ego_arr, others_arr)
        dangerous_idx = int(np.argmin(v_at_ego))

        # Plot the BRT slice for the most-dangerous vehicle only.
        values = bu.query_slice_single(
            dynamics, vf_model, px_grid, py_grid,
            ego_psi=ego.psi, ego_v=ego.v, other=others_arr[dangerous_idx],
        )
        frames_values.append(values)
        title_suffix_per_frame.append(
            f"most dangerous = #{dangerous_idx} (V={v_at_ego[dangerous_idx]:.1f})"
        )

    raw_vbar = float(np.max(np.abs(np.concatenate([v.flatten() for v in frames_values]))))
    print(f"  raw value magnitude range: ±{raw_vbar:.2f}; clipping colormap to ±{bu.VALUE_CMAP_LIMIT}")

    bu.VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    bu.save_sidebyside_video(
        SIDEBYSIDE_OUTPUT_PATH, rollout_frames,
        px_axis, py_axis, frames_values,
        ego_xy, others_xy_per_frame, n_frames,
        title_suffix_per_frame=title_suffix_per_frame,
    )
    print("Done.")


if __name__ == "__main__":
    main()
