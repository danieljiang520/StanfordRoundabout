"""Visualize the TwoCar8D BRT over a 2-car roundabout rollout.

Runs one episode of the DQN-GNN policy in VariableRoundabout-v0 with one
traffic vehicle, then animates a (px1, py1) slice of the trained BRT —
re-sliced each frame at the live (psi1, v1, px2, py2, psi2, v2) — with a
green dot tracking the ego and a blue dot tracking the other vehicle.
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from stable_baselines3 import DQN
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "libraries" / "DeepReach_MPC"))

# noqa: E402 — sys.path manipulation must precede these imports.
import src  # registers VariableRoundabout-v0  # noqa: E402
from libraries.DeepReach_MPC.dynamics.dynamics import TwoCar8D  # noqa: E402
from libraries.DeepReach_MPC.utils import modules  # noqa: E402


CKPT_PATH = REPO_ROOT / "vf_30k_epoch.ckpt"
MODEL_PATH = REPO_ROOT / "roundabout_dqn_gnn" / "model"
OUTPUT_PATH = REPO_ROOT / "roundabout_dqn_gnn" / "videos" / "brt_overlay.gif"

# TwoCar8D params from train_TwoCar.py (the run that produced vf_30k_epoch.ckpt).
COLLISION_R = 1.0
WHEELBASE = 5.0
T_QUERY = 1.0  # BRT time horizon — same as problem3_helper.NeuralVF.values.

# Plot extent in world frame. The roundabout sits within ~30 m of origin,
# but the ego exits along the south road and quickly leaves that window, so
# widen the view to track the full trajectory.
GRID_EXTENT = 60.0
GRID_RESOLUTION = 100

# The BRT boundary function (px1-px2)^2 + (py1-py2)^2 - collisionR^2 dominates
# the network output and can reach ~10^4 on this grid. The interesting
# information (sign change and shape near the boundary) lives in a much
# smaller range, so clip the colormap aggressively.
VALUE_CMAP_LIMIT = 50.0

VEHICLES_COUNT_OBS = 12
TRAFFIC_VEHICLES = 1
EPISODE_DURATION = 20  # roundabout default is 11; modest extension.
SEED_CANDIDATES = list(range(20))  # try these and keep the longest non-immediate-crash run.

ROUNDABOUT_CONFIG = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": VEHICLES_COUNT_OBS,
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": True,
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-15, 15],
            "vy": [-15, 15],
        },
    },
    "traffic_vehicles_count": TRAFFIC_VEHICLES,
    "duration": EPISODE_DURATION,
}


def _build_cpu_dynamics() -> TwoCar8D:
    """Construct TwoCar8D and move its hard-coded CUDA buffers to CPU.

    dynamics.py was written for training on a GPU; we only need CPU inference.
    """
    # TwoCar8D.__init__ calls .cuda() on its tensor buffers — temporarily
    # remap cuda() to cpu() while constructing, so the call doesn't crash on
    # a machine without CUDA.
    original_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self.cpu()  # type: ignore[method-assign]
    try:
        dynamics = TwoCar8D(collisionR=COLLISION_R, wheelbase=WHEELBASE, set_mode="avoid")
    finally:
        torch.Tensor.cuda = original_cuda  # type: ignore[method-assign]

    # periodic_transform_fn returns .cuda() at the end — replace with a CPU
    # version. Bind state_var on the instance for the sin/cos scaling.
    state_var_cpu = dynamics.state_var.detach().cpu()

    def periodic_transform_fn_cpu(input_tensor: torch.Tensor) -> torch.Tensor:
        output_shape = list(input_tensor.shape)
        output_shape[-1] = input_tensor.shape[-1] + 2
        transformed = torch.zeros(output_shape, device=input_tensor.device)
        sv = state_var_cpu.to(input_tensor.device)
        transformed[..., 0:3] = input_tensor[..., 0:3]
        transformed[..., 3] = torch.sin(input_tensor[..., 3] * sv[2])
        transformed[..., 4] = torch.cos(input_tensor[..., 3] * sv[2])
        transformed[..., 5:8] = input_tensor[..., 4:7]
        transformed[..., 8] = torch.sin(input_tensor[..., 7] * sv[6])
        transformed[..., 9] = torch.cos(input_tensor[..., 7] * sv[6])
        transformed[..., 10] = input_tensor[..., 8]
        return transformed

    dynamics.periodic_transform_fn = periodic_transform_fn_cpu  # type: ignore[assignment]
    return dynamics


def _load_value_network(dynamics: TwoCar8D) -> torch.nn.Module:
    model = modules.SingleBVPNet(
        in_features=dynamics.input_dim,
        out_features=1,
        type="sine",
        mode="mlp",
        final_layer_factor=1.0,
        hidden_features=512,
        num_hidden_layers=3,
        periodic_transform_fn=dynamics.periodic_transform_fn,
    )
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def _query_slice(
    dynamics: TwoCar8D,
    model: torch.nn.Module,
    px_grid: np.ndarray,
    py_grid: np.ndarray,
    ego_psi: float,
    ego_v: float,
    other: np.ndarray,
) -> np.ndarray:
    """Return values on the (px1, py1) grid at the given fixed-dim state."""
    nx, ny = px_grid.shape
    n = nx * ny
    px1 = torch.from_numpy(px_grid.reshape(-1)).float()
    py1 = torch.from_numpy(py_grid.reshape(-1)).float()
    psi1 = torch.full((n,), ego_psi, dtype=torch.float32)
    v1 = torch.full((n,), ego_v, dtype=torch.float32)
    px2 = torch.full((n,), float(other[0]), dtype=torch.float32)
    py2 = torch.full((n,), float(other[1]), dtype=torch.float32)
    psi2 = torch.full((n,), float(other[2]), dtype=torch.float32)
    v2 = torch.full((n,), float(other[3]), dtype=torch.float32)
    t = torch.full((n, 1), T_QUERY, dtype=torch.float32)

    state = torch.stack([px1, py1, psi1, v1, px2, py2, psi2, v2], dim=-1)
    coords = torch.cat([t, state], dim=-1)
    model_input = dynamics.coord_to_input(coords)
    with torch.no_grad():
        results = model({"coords": model_input})
    values = dynamics.io_to_value(results["model_in"].detach(), results["model_out"].detach().squeeze(dim=-1))
    return values.cpu().numpy().reshape(nx, ny)


def main():
    print("Loading DeepReach BRT...")
    dynamics = _build_cpu_dynamics()
    vf_model = _load_value_network(dynamics)

    print(f"Rolling out DQN-GNN policy with {TRAFFIC_VEHICLES} traffic vehicle(s)...")
    env = gym.make("VariableRoundabout-v0", render_mode=None, config=ROUNDABOUT_CONFIG)
    policy = DQN.load(str(MODEL_PATH), env=env)

    # Many seeds produce an immediate-crash rollout (2 frames). Try several
    # seeds and keep the longest one so the GIF actually shows BRT evolution.
    best_rollout = None
    for seed in SEED_CANDIDATES:
        candidate = src.rollout_with_states(env, policy, deterministic=True, seed=seed)
        n = len(candidate["states"])
        print(f"  seed={seed}: {n} frames, crashed={candidate['crashed']}")
        if best_rollout is None or n > len(best_rollout["states"]):
            best_rollout = candidate
    rollout = best_rollout
    env.close()
    n_frames = len(rollout["states"])
    print(f"  picked rollout: {n_frames} frames, crashed={rollout['crashed']}")
    if n_frames < 2:
        print("  WARNING: rollout terminated immediately; the GIF will be very short.")

    # Build the (px1, py1) grid once.
    px_axis = np.linspace(-GRID_EXTENT, GRID_EXTENT, GRID_RESOLUTION, dtype=np.float32)
    py_axis = np.linspace(-GRID_EXTENT, GRID_EXTENT, GRID_RESOLUTION, dtype=np.float32)
    px_grid, py_grid = np.meshgrid(px_axis, py_axis, indexing="ij")

    # Pre-compute values for every frame so the GIF write phase is just IO.
    print("Querying BRT along trajectory...")
    frames_values: list[np.ndarray] = []
    ego_xy: list[tuple[float, float]] = []
    other_xy: list[tuple[float, float]] = []
    for states in tqdm(rollout["states"]):
        ego = states[0]
        other = states[1] if len(states) > 1 else states[0]  # safety fallback
        values = _query_slice(
            dynamics,
            vf_model,
            px_grid,
            py_grid,
            ego_psi=ego.psi,
            ego_v=ego.v,
            other=other.as_array(),
        )
        frames_values.append(values)
        ego_xy.append((ego.px, ego.py))
        other_xy.append((other.px, other.py))

    raw_vbar = float(np.max(np.abs(np.concatenate([v.flatten() for v in frames_values]))))
    print(f"  raw value magnitude range: ±{raw_vbar:.2f}; clipping colormap to ±{VALUE_CMAP_LIMIT}")
    vbar = VALUE_CMAP_LIMIT

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_xlim(-GRID_EXTENT, GRID_EXTENT)
    ax.set_ylim(-GRID_EXTENT, GRID_EXTENT)
    title = ax.set_title("BRT slice — frame 0")

    mesh = ax.pcolormesh(
        px_axis,
        py_axis,
        frames_values[0].T,
        cmap="RdBu",
        vmin=-vbar,
        vmax=+vbar,
        shading="auto",
    )
    plt.colorbar(mesh, ax=ax, label="V(x, t=1)")

    contour_state = {"artist": ax.contour(px_axis, py_axis, frames_values[0].T, levels=[0], colors="k")}
    ego_dot, = ax.plot([ego_xy[0][0]], [ego_xy[0][1]], "o", color="lime", markersize=10, markeredgecolor="black", label="ego")
    other_dot, = ax.plot([other_xy[0][0]], [other_xy[0][1]], "o", color="blue", markersize=10, markeredgecolor="black", label="other")
    ax.legend(loc="upper right")

    def update(i):
        title.set_text(f"BRT slice — frame {i} / {n_frames - 1}")
        mesh.set_array(frames_values[i].T.ravel())
        contour_state["artist"].remove()
        contour_state["artist"] = ax.contour(px_axis, py_axis, frames_values[i].T, levels=[0], colors="k")
        ego_dot.set_data([ego_xy[i][0]], [ego_xy[i][1]])
        other_dot.set_data([other_xy[i][0]], [other_xy[i][1]])
        return mesh, ego_dot, other_dot

    anim = FuncAnimation(fig=fig, func=update, frames=np.arange(n_frames), interval=200)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving GIF to {OUTPUT_PATH}...")
    with tqdm(total=n_frames) as pbar:
        anim.save(
            filename=str(OUTPUT_PATH),
            writer="pillow",
            progress_callback=lambda i, n: pbar.update(1),
        )
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
