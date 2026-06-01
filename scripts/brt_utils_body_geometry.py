# """Shared helpers for TwoCar8D BRT rollout visualizations.

# Used by:
# - scripts/visualize_brt_rollout.py        (1 other vehicle)
# - scripts/visualize_brt_rollout_multicar.py (N other vehicles, value = min over them)

# The min-over-vehicles trick follows the multi-obstacle pattern in
# homework/AA276_GCP/hw3/problem3.py: query the value function once per other
# vehicle and take the elementwise minimum, since each per-vehicle V is the
# value of the safety problem "ego vs. that one vehicle". No coordinate shift
# is needed because TwoCar8D's V takes both cars' full kinematic states.
# """

# from __future__ import annotations

# import sys
# from pathlib import Path

# import matplotlib.pyplot as plt
# import numpy as np
# import torch
# from matplotlib.animation import FuncAnimation
# from tqdm import tqdm

# REPO_ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(REPO_ROOT))
# sys.path.insert(0, str(REPO_ROOT / "libraries" / "DeepReach_MPC"))

# # noqa: E402 — sys.path manipulation must precede these imports.
# from libraries.DeepReach_MPC.dynamics.dynamics import TwoCar8D  # noqa: E402
# from libraries.DeepReach_MPC.utils import modules  # noqa: E402


# CKPT_PATH = REPO_ROOT / "vf_30k_epoch.ckpt"
# MODEL_PATH = REPO_ROOT / "roundabout_dqn_gnn" / "model"
# VIDEOS_DIR = REPO_ROOT / "roundabout_dqn_gnn" / "videos"

# # TwoCar8D body-geometry params. Retrain before treating CKPT_PATH as body-geometry BRT.
# CAR_LENGTH = 5.0
# CAR_WIDTH = 2.0
# BODY_MARGIN = 1.0
# WHEELBASE = 5.0
# T_QUERY = 1.0

# # Plot extent in world frame. The roundabout sits within ~30 m of origin, but
# # vehicles exit along the radial roads, so widen the view to keep them onscreen.
# GRID_EXTENT = 60.0
# GRID_RESOLUTION = 100

# # The body-geometry BRT boundary is a signed oriented-rectangle separation in
# # meters. Clip the colormap to keep the near-zero/contact structure visible.
# VALUE_CMAP_LIMIT = 50.0

# VIDEO_FPS = 5

# VEHICLES_COUNT_OBS = 12
# EPISODE_DURATION = 20

# # Standard observation+env config matched to the DQN-GNN training script.
# # Callers override "traffic_vehicles_count" per script.
# ROUNDABOUT_CONFIG_BASE = {
#     # RGB frames are consumed by matplotlib/video export; no interactive pygame
#     # window is needed, and closing one mid-rollout breaks highway-env rendering.
#     "offscreen_rendering": True,
#     "observation": {
#         "type": "Kinematics",
#         "vehicles_count": VEHICLES_COUNT_OBS,
#         "features": ["presence", "x", "y", "vx", "vy"],
#         "absolute": True,
#         "features_range": {
#             "x": [-100, 100],
#             "y": [-100, 100],
#             "vx": [-15, 15],
#             "vy": [-15, 15],
#         },
#     },
#     "duration": EPISODE_DURATION,
# }


# def make_roundabout_config(traffic_vehicles: int) -> dict:
#     cfg = {**ROUNDABOUT_CONFIG_BASE}
#     cfg["observation"] = {**ROUNDABOUT_CONFIG_BASE["observation"]}
#     cfg["traffic_vehicles_count"] = traffic_vehicles
#     return cfg


# def build_cpu_dynamics() -> TwoCar8D:
#     """Construct TwoCar8D and rewire its hard-coded CUDA buffers for CPU inference."""
#     # TwoCar8D.__init__ calls .cuda() on its tensor buffers. Temporarily remap
#     # cuda() to cpu() while constructing, so the call doesn't crash on a
#     # machine without CUDA.
#     original_cuda = torch.Tensor.cuda
#     torch.Tensor.cuda = lambda self, *a, **k: self.cpu()  # type: ignore[method-assign]
#     try:
#         dynamics = TwoCar8D(
#             car_length=CAR_LENGTH,
#             car_width=CAR_WIDTH,
#             body_margin=BODY_MARGIN,
#             wheelbase=WHEELBASE,
#             set_mode="avoid",
#         )
#     finally:
#         torch.Tensor.cuda = original_cuda  # type: ignore[method-assign]

#     state_var_cpu = dynamics.state_var.detach().cpu()

#     def periodic_transform_fn_cpu(input_tensor: torch.Tensor) -> torch.Tensor:
#         output_shape = list(input_tensor.shape)
#         output_shape[-1] = input_tensor.shape[-1] + 2
#         transformed = torch.zeros(output_shape, device=input_tensor.device)
#         sv = state_var_cpu.to(input_tensor.device)
#         transformed[..., 0:3] = input_tensor[..., 0:3]
#         transformed[..., 3] = torch.sin(input_tensor[..., 3] * sv[2])
#         transformed[..., 4] = torch.cos(input_tensor[..., 3] * sv[2])
#         transformed[..., 5:8] = input_tensor[..., 4:7]
#         transformed[..., 8] = torch.sin(input_tensor[..., 7] * sv[6])
#         transformed[..., 9] = torch.cos(input_tensor[..., 7] * sv[6])
#         transformed[..., 10] = input_tensor[..., 8]
#         return transformed

#     dynamics.periodic_transform_fn = periodic_transform_fn_cpu  # type: ignore[assignment]
#     return dynamics


# def load_value_network(dynamics: TwoCar8D) -> torch.nn.Module:
#     model = modules.SingleBVPNet(
#         in_features=dynamics.input_dim,
#         out_features=1,
#         type="sine",
#         mode="mlp",
#         final_layer_factor=1.0,
#         hidden_features=512,
#         num_hidden_layers=3,
#         periodic_transform_fn=dynamics.periodic_transform_fn,
#     )
#     ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
#     model.load_state_dict(ckpt["model"])
#     model.eval()
#     return model


# def make_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     px_axis = np.linspace(-GRID_EXTENT, GRID_EXTENT, GRID_RESOLUTION, dtype=np.float32)
#     py_axis = np.linspace(-GRID_EXTENT, GRID_EXTENT, GRID_RESOLUTION, dtype=np.float32)
#     px_grid, py_grid = np.meshgrid(px_axis, py_axis, indexing="ij")
#     return px_axis, py_axis, px_grid, py_grid


# def query_slice_single(
#     dynamics: TwoCar8D,
#     model: torch.nn.Module,
#     px_grid: np.ndarray,
#     py_grid: np.ndarray,
#     ego_psi: float,
#     ego_v: float,
#     other: np.ndarray,
# ) -> np.ndarray:
#     """Return V on the (px1, py1) grid for a single other vehicle."""
#     nx, ny = px_grid.shape
#     n = nx * ny
#     px1 = torch.from_numpy(px_grid.reshape(-1)).float()
#     py1 = torch.from_numpy(py_grid.reshape(-1)).float()
#     psi1 = torch.full((n,), ego_psi, dtype=torch.float32)
#     v1 = torch.full((n,), ego_v, dtype=torch.float32)
#     px2 = torch.full((n,), float(other[0]), dtype=torch.float32)
#     py2 = torch.full((n,), float(other[1]), dtype=torch.float32)
#     psi2 = torch.full((n,), float(other[2]), dtype=torch.float32)
#     v2 = torch.full((n,), float(other[3]), dtype=torch.float32)
#     t = torch.full((n, 1), T_QUERY, dtype=torch.float32)

#     state = torch.stack([px1, py1, psi1, v1, px2, py2, psi2, v2], dim=-1)
#     coords = torch.cat([t, state], dim=-1)
#     model_input = dynamics.coord_to_input(coords)
#     with torch.no_grad():
#         results = model({"coords": model_input})
#     values = dynamics.io_to_value(
#         results["model_in"].detach(), results["model_out"].detach().squeeze(dim=-1)
#     )
#     return values.cpu().numpy().reshape(nx, ny)


# def query_slice_multi(
#     dynamics: TwoCar8D,
#     model: torch.nn.Module,
#     px_grid: np.ndarray,
#     py_grid: np.ndarray,
#     ego_psi: float,
#     ego_v: float,
#     others: np.ndarray,
# ) -> tuple[np.ndarray, np.ndarray]:
#     """Return (min-V over K vehicles, argmin index map) on the (px1, py1) grid.

#     others: (K, 4) array of [px2, py2, psi2, v2] for each other vehicle.

#     Mirrors the multi-obstacle min in problem3.py: V_min(x) = min_i V_i(x).
#     The argmin map names the dominating vehicle at each grid cell.
#     """
#     K = others.shape[0]
#     nx, ny = px_grid.shape
#     n = nx * ny

#     # Per-vehicle (K) × per-grid-cell (n) batch dimension.
#     px1 = torch.from_numpy(px_grid.reshape(-1)).float().repeat(K)             # (K*n,)
#     py1 = torch.from_numpy(py_grid.reshape(-1)).float().repeat(K)
#     psi1 = torch.full((K * n,), ego_psi, dtype=torch.float32)
#     v1 = torch.full((K * n,), ego_v, dtype=torch.float32)
#     # Repeat each other-vehicle's scalar across the n grid cells.
#     others_t = torch.from_numpy(others).float()                               # (K, 4)
#     px2 = others_t[:, 0].repeat_interleave(n)
#     py2 = others_t[:, 1].repeat_interleave(n)
#     psi2 = others_t[:, 2].repeat_interleave(n)
#     v2 = others_t[:, 3].repeat_interleave(n)
#     t = torch.full((K * n, 1), T_QUERY, dtype=torch.float32)

#     state = torch.stack([px1, py1, psi1, v1, px2, py2, psi2, v2], dim=-1)
#     coords = torch.cat([t, state], dim=-1)
#     model_input = dynamics.coord_to_input(coords)
#     with torch.no_grad():
#         results = model({"coords": model_input})
#     values = dynamics.io_to_value(
#         results["model_in"].detach(), results["model_out"].detach().squeeze(dim=-1)
#     )
#     per_vehicle = values.cpu().numpy().reshape(K, nx, ny)
#     min_values = per_vehicle.min(axis=0)
#     argmin_map = per_vehicle.argmin(axis=0)
#     return min_values, argmin_map


# def nearest_grid_cell(px_axis: np.ndarray, py_axis: np.ndarray, x: float, y: float) -> tuple[int, int]:
#     ix = int(np.clip(np.searchsorted(px_axis, x), 0, len(px_axis) - 1))
#     iy = int(np.clip(np.searchsorted(py_axis, y), 0, len(py_axis) - 1))
#     return ix, iy


# def decorate_brt_axes(ax) -> None:
#     """Set up BRT axes with inverted y (negative on top, positive on bottom)."""
#     ax.set_xlabel("x (m)")
#     ax.set_ylabel("y (m)")
#     ax.set_aspect("equal")
#     ax.set_xlim(-GRID_EXTENT, GRID_EXTENT)
#     ax.set_ylim(GRID_EXTENT, -GRID_EXTENT)


# def save_brt_video(
#     path,
#     px_axis,
#     py_axis,
#     frames_values,
#     ego_xy,
#     others_xy_per_frame,
#     n_frames,
#     vbar=VALUE_CMAP_LIMIT,
#     title_suffix_per_frame=None,
# ):
#     """Animate the BRT heatmap with ego (green) and one or more others (blue) dots.

#     others_xy_per_frame: list of length n_frames, each element a list of (x, y)
#         tuples — one per present other vehicle on that frame. The number can
#         vary across frames; the dot count adjusts via scatter offsets.
#     title_suffix_per_frame: optional list[str] appended to each frame's title.
#     """
#     print(f"Saving BRT video to {path}...")
#     fig, ax = plt.subplots(figsize=(6, 6))
#     decorate_brt_axes(ax)
#     title = ax.set_title("BRT slice — frame 0")

#     mesh = ax.pcolormesh(
#         px_axis, py_axis, frames_values[0].T,
#         cmap="RdBu", vmin=-vbar, vmax=+vbar, shading="auto",
#     )
#     plt.colorbar(mesh, ax=ax, label="V(x, t=1)")

#     contour_state = {"artist": ax.contour(px_axis, py_axis, frames_values[0].T, levels=[0], colors="k")}
#     ego_dot, = ax.plot([ego_xy[0][0]], [ego_xy[0][1]], "o", color="lime",
#                        markersize=10, markeredgecolor="black", label="ego")
#     others0 = np.array(others_xy_per_frame[0]) if others_xy_per_frame[0] else np.zeros((0, 2))
#     others_scatter = ax.scatter(
#         others0[:, 0] if len(others0) else [],
#         others0[:, 1] if len(others0) else [],
#         s=80, c="blue", edgecolors="black", label="other",
#     )
#     ax.legend(loc="upper right")

#     def update(i):
#         suffix = f" — {title_suffix_per_frame[i]}" if title_suffix_per_frame else ""
#         title.set_text(f"BRT slice — frame {i} / {n_frames - 1}{suffix}")
#         mesh.set_array(frames_values[i].T.ravel())
#         contour_state["artist"].remove()
#         contour_state["artist"] = ax.contour(px_axis, py_axis, frames_values[i].T, levels=[0], colors="k")
#         ego_dot.set_data([ego_xy[i][0]], [ego_xy[i][1]])
#         pts = np.array(others_xy_per_frame[i]) if others_xy_per_frame[i] else np.zeros((0, 2))
#         others_scatter.set_offsets(pts)
#         return mesh, ego_dot, others_scatter

#     anim = FuncAnimation(fig=fig, func=update, frames=np.arange(n_frames), interval=200)
#     with tqdm(total=n_frames) as pbar:
#         anim.save(filename=str(path), writer="ffmpeg", fps=VIDEO_FPS,
#                   progress_callback=lambda i, n: pbar.update(1))
#     plt.close(fig)


# def save_rollout_video(path, rollout_frames):
#     print(f"Saving rollout video to {path}...")
#     n = len(rollout_frames)
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.set_axis_off()
#     title = ax.set_title("Rollout — frame 0")
#     img = ax.imshow(rollout_frames[0])

#     def update(i):
#         title.set_text(f"Rollout — frame {i} / {n - 1}")
#         img.set_data(rollout_frames[i])
#         return (img,)

#     anim = FuncAnimation(fig=fig, func=update, frames=np.arange(n), interval=200)
#     with tqdm(total=n) as pbar:
#         anim.save(filename=str(path), writer="ffmpeg", fps=VIDEO_FPS,
#                   progress_callback=lambda i, _n: pbar.update(1))
#     plt.close(fig)


# def save_sidebyside_video(
#     path,
#     rollout_frames,
#     px_axis,
#     py_axis,
#     frames_values,
#     ego_xy,
#     others_xy_per_frame,
#     n_frames,
#     vbar=VALUE_CMAP_LIMIT,
#     title_suffix_per_frame=None,
# ):
#     print(f"Saving side-by-side video to {path}...")
#     n = min(len(rollout_frames), n_frames)
#     fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

#     ax_left.set_axis_off()
#     left_title = ax_left.set_title("Rollout — frame 0")
#     img = ax_left.imshow(rollout_frames[0])

#     decorate_brt_axes(ax_right)
#     right_title = ax_right.set_title("BRT slice — frame 0")
#     mesh = ax_right.pcolormesh(
#         px_axis, py_axis, frames_values[0].T,
#         cmap="RdBu", vmin=-vbar, vmax=+vbar, shading="auto",
#     )
#     plt.colorbar(mesh, ax=ax_right, label="V(x, t=1)")
#     contour_state = {"artist": ax_right.contour(px_axis, py_axis, frames_values[0].T, levels=[0], colors="k")}
#     ego_dot, = ax_right.plot([ego_xy[0][0]], [ego_xy[0][1]], "o", color="lime",
#                              markersize=10, markeredgecolor="black", label="ego")
#     others0 = np.array(others_xy_per_frame[0]) if others_xy_per_frame[0] else np.zeros((0, 2))
#     others_scatter = ax_right.scatter(
#         others0[:, 0] if len(others0) else [],
#         others0[:, 1] if len(others0) else [],
#         s=80, c="blue", edgecolors="black", label="other",
#     )
#     ax_right.legend(loc="upper right")

#     def update(i):
#         left_title.set_text(f"Rollout — frame {i} / {n - 1}")
#         img.set_data(rollout_frames[i])
#         suffix = f" — {title_suffix_per_frame[i]}" if title_suffix_per_frame else ""
#         right_title.set_text(f"BRT slice — frame {i} / {n - 1}{suffix}")
#         mesh.set_array(frames_values[i].T.ravel())
#         contour_state["artist"].remove()
#         contour_state["artist"] = ax_right.contour(px_axis, py_axis, frames_values[i].T, levels=[0], colors="k")
#         ego_dot.set_data([ego_xy[i][0]], [ego_xy[i][1]])
#         pts = np.array(others_xy_per_frame[i]) if others_xy_per_frame[i] else np.zeros((0, 2))
#         others_scatter.set_offsets(pts)
#         return img, mesh, ego_dot, others_scatter

#     anim = FuncAnimation(fig=fig, func=update, frames=np.arange(n), interval=200)
#     with tqdm(total=n) as pbar:
#         anim.save(filename=str(path), writer="ffmpeg", fps=VIDEO_FPS,
#                   progress_callback=lambda i, _n: pbar.update(1))
#     plt.close(fig)


##### Lastest version
"""Shared helpers for TwoCar8D BRT rollout visualizations.

Used by:
- scripts/visualize_brt_rollout.py        (1 other vehicle)
- scripts/visualize_brt_rollout_multicar.py (N other vehicles, value = min over them)

The min-over-vehicles trick follows the multi-obstacle pattern in
homework/AA276_GCP/hw3/problem3.py: query the value function once per other
vehicle and take the elementwise minimum, since each per-vehicle V is the
value of the safety problem "ego vs. that one vehicle". No coordinate shift
is needed because TwoCar8D's V takes both cars' full kinematic states.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "libraries" / "DeepReach_MPC"))

# noqa: E402 — sys.path manipulation must precede these imports.
from libraries.DeepReach_MPC.dynamics.dynamics import TwoCar8D  # noqa: E402
from libraries.DeepReach_MPC.utils import modules  # noqa: E402


CKPT_PATH = REPO_ROOT / "vf_150k.ckpt"
MODEL_PATH = REPO_ROOT / "roundabout_dqn_gnn" / "model"
VIDEOS_DIR = REPO_ROOT / "roundabout_dqn_gnn" / "videos"

# TwoCar8D params for body-geometry training/evaluation. Retrain the value
# function before treating CKPT_PATH as a body-geometry BRT checkpoint.
CAR_LENGTH = 5.0
CAR_WIDTH = 2.0
BODY_MARGIN = 1.0
WHEELBASE = 5.0
T_QUERY = 1.0

# Plot extent in world frame. The roundabout sits within ~30 m of origin, but
# vehicles exit along the radial roads, so widen the view to keep them onscreen.
GRID_EXTENT = 60.0
GRID_RESOLUTION = 100

# The body-geometry BRT boundary is a signed oriented-rectangle separation in
# meters. Clip the colormap to keep the near-zero/contact structure visible.
VALUE_CMAP_LIMIT = 50.0

VIDEO_FPS = 5

VEHICLES_COUNT_OBS = 12
EPISODE_DURATION = 20

# Standard observation+env config matched to the DQN-GNN training script.
# Callers override "traffic_vehicles_count" per script.
ROUNDABOUT_CONFIG_BASE = {
    # Video scripts consume RGB arrays and do not need an interactive pygame window.
    "offscreen_rendering": True,
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
    "duration": EPISODE_DURATION,
}


def make_roundabout_config(traffic_vehicles: int) -> dict:
    cfg = {**ROUNDABOUT_CONFIG_BASE}
    cfg["observation"] = {**ROUNDABOUT_CONFIG_BASE["observation"]}
    cfg["traffic_vehicles_count"] = traffic_vehicles
    return cfg


def build_cpu_dynamics() -> TwoCar8D:
    """Construct TwoCar8D and rewire its hard-coded CUDA buffers for CPU inference."""
    # TwoCar8D.__init__ calls .cuda() on its tensor buffers. Temporarily remap
    # cuda() to cpu() while constructing, so the call doesn't crash on a
    # machine without CUDA.
    original_cuda = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *a, **k: self.cpu()  # type: ignore[method-assign]
    try:
        dynamics = TwoCar8D(
            car_length=CAR_LENGTH,
            car_width=CAR_WIDTH,
            body_margin=BODY_MARGIN,
            wheelbase=WHEELBASE,
            set_mode="avoid",
        )
    finally:
        torch.Tensor.cuda = original_cuda  # type: ignore[method-assign]

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


def load_value_network(dynamics: TwoCar8D) -> torch.nn.Module:
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


def make_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px_axis = np.linspace(-GRID_EXTENT, GRID_EXTENT, GRID_RESOLUTION, dtype=np.float32)
    py_axis = np.linspace(-GRID_EXTENT, GRID_EXTENT, GRID_RESOLUTION, dtype=np.float32)
    px_grid, py_grid = np.meshgrid(px_axis, py_axis, indexing="ij")
    return px_axis, py_axis, px_grid, py_grid


def query_slice_single(
    dynamics: TwoCar8D,
    model: torch.nn.Module,
    px_grid: np.ndarray,
    py_grid: np.ndarray,
    ego_psi: float,
    ego_v: float,
    other: np.ndarray,
) -> np.ndarray:
    """Return V on the (px1, py1) grid for a single other vehicle."""
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
    values = dynamics.io_to_value(
        results["model_in"].detach(), results["model_out"].detach().squeeze(dim=-1)
    )
    return values.cpu().numpy().reshape(nx, ny)


def query_slice_multi(
    dynamics: TwoCar8D,
    model: torch.nn.Module,
    px_grid: np.ndarray,
    py_grid: np.ndarray,
    ego_psi: float,
    ego_v: float,
    others: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (min-V over K vehicles, argmin index map) on the (px1, py1) grid.

    others: (K, 4) array of [px2, py2, psi2, v2] for each other vehicle.

    Mirrors the multi-obstacle min in problem3.py: V_min(x) = min_i V_i(x).
    The argmin map names the dominating vehicle at each grid cell.
    """
    K = others.shape[0]
    nx, ny = px_grid.shape
    n = nx * ny

    # Per-vehicle (K) × per-grid-cell (n) batch dimension.
    px1 = torch.from_numpy(px_grid.reshape(-1)).float().repeat(K)             # (K*n,)
    py1 = torch.from_numpy(py_grid.reshape(-1)).float().repeat(K)
    psi1 = torch.full((K * n,), ego_psi, dtype=torch.float32)
    v1 = torch.full((K * n,), ego_v, dtype=torch.float32)
    # Repeat each other-vehicle's scalar across the n grid cells.
    others_t = torch.from_numpy(others).float()                               # (K, 4)
    px2 = others_t[:, 0].repeat_interleave(n)
    py2 = others_t[:, 1].repeat_interleave(n)
    psi2 = others_t[:, 2].repeat_interleave(n)
    v2 = others_t[:, 3].repeat_interleave(n)
    t = torch.full((K * n, 1), T_QUERY, dtype=torch.float32)

    state = torch.stack([px1, py1, psi1, v1, px2, py2, psi2, v2], dim=-1)
    coords = torch.cat([t, state], dim=-1)
    model_input = dynamics.coord_to_input(coords)
    with torch.no_grad():
        results = model({"coords": model_input})
    values = dynamics.io_to_value(
        results["model_in"].detach(), results["model_out"].detach().squeeze(dim=-1)
    )
    per_vehicle = values.cpu().numpy().reshape(K, nx, ny)
    min_values = per_vehicle.min(axis=0)
    argmin_map = per_vehicle.argmin(axis=0)
    return min_values, argmin_map


def values_at_ego_state(
    dynamics: TwoCar8D,
    model: torch.nn.Module,
    ego_state: np.ndarray,
    others: np.ndarray,
) -> np.ndarray:
    """Evaluate V at the ego's actual state, once per other vehicle.

    Returns a length-K array of V_i(ego_state, other_i). Useful for picking
    the most-threatening other vehicle (argmin) at the current world state.

    ego_state: (4,) array [px1, py1, psi1, v1].
    others:    (K, 4) array of [px2, py2, psi2, v2].
    """
    K = others.shape[0]
    ego_t = torch.from_numpy(np.asarray(ego_state, dtype=np.float32))         # (4,)
    others_t = torch.from_numpy(np.asarray(others, dtype=np.float32))         # (K, 4)
    ego_block = ego_t.unsqueeze(0).expand(K, 4)                               # (K, 4)
    state = torch.cat([ego_block, others_t], dim=-1)                          # (K, 8)
    t = torch.full((K, 1), T_QUERY, dtype=torch.float32)
    coords = torch.cat([t, state], dim=-1)
    model_input = dynamics.coord_to_input(coords)
    with torch.no_grad():
        results = model({"coords": model_input})
    values = dynamics.io_to_value(
        results["model_in"].detach(), results["model_out"].detach().squeeze(dim=-1)
    )
    return values.cpu().numpy()


def nearest_grid_cell(px_axis: np.ndarray, py_axis: np.ndarray, x: float, y: float) -> tuple[int, int]:
    ix = int(np.clip(np.searchsorted(px_axis, x), 0, len(px_axis) - 1))
    iy = int(np.clip(np.searchsorted(py_axis, y), 0, len(py_axis) - 1))
    return ix, iy


def decorate_brt_axes(ax) -> None:
    """Set up BRT axes with inverted y (negative on top, positive on bottom)."""
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.set_xlim(-GRID_EXTENT, GRID_EXTENT)
    ax.set_ylim(GRID_EXTENT, -GRID_EXTENT)


def save_brt_video(
    path,
    px_axis,
    py_axis,
    frames_values,
    ego_xy,
    others_xy_per_frame,
    n_frames,
    vbar=VALUE_CMAP_LIMIT,
    title_suffix_per_frame=None,
):
    """Animate the BRT heatmap with ego (green) and one or more others (blue) dots.

    others_xy_per_frame: list of length n_frames, each element a list of (x, y)
        tuples — one per present other vehicle on that frame. The number can
        vary across frames; the dot count adjusts via scatter offsets.
    title_suffix_per_frame: optional list[str] appended to each frame's title.
    """
    print(f"Saving BRT video to {path}...")
    fig, ax = plt.subplots(figsize=(6, 6))
    decorate_brt_axes(ax)
    title = ax.set_title("BRT slice — frame 0")

    mesh = ax.pcolormesh(
        px_axis, py_axis, frames_values[0].T,
        cmap="RdBu", vmin=-vbar, vmax=+vbar, shading="auto",
    )
    plt.colorbar(mesh, ax=ax, label="V(x, t=1)")

    contour_state = {"artist": ax.contour(px_axis, py_axis, frames_values[0].T, levels=[0], colors="k")}
    ego_dot, = ax.plot([ego_xy[0][0]], [ego_xy[0][1]], "o", color="lime",
                       markersize=10, markeredgecolor="black", label="ego")
    others0 = np.array(others_xy_per_frame[0]) if others_xy_per_frame[0] else np.zeros((0, 2))
    others_scatter = ax.scatter(
        others0[:, 0] if len(others0) else [],
        others0[:, 1] if len(others0) else [],
        s=80, c="blue", edgecolors="black", label="other",
    )
    ax.legend(loc="upper right")

    def update(i):
        suffix = f" — {title_suffix_per_frame[i]}" if title_suffix_per_frame else ""
        title.set_text(f"BRT slice — frame {i} / {n_frames - 1}{suffix}")
        mesh.set_array(frames_values[i].T.ravel())
        contour_state["artist"].remove()
        contour_state["artist"] = ax.contour(px_axis, py_axis, frames_values[i].T, levels=[0], colors="k")
        ego_dot.set_data([ego_xy[i][0]], [ego_xy[i][1]])
        pts = np.array(others_xy_per_frame[i]) if others_xy_per_frame[i] else np.zeros((0, 2))
        others_scatter.set_offsets(pts)
        return mesh, ego_dot, others_scatter

    anim = FuncAnimation(fig=fig, func=update, frames=np.arange(n_frames), interval=200)
    with tqdm(total=n_frames) as pbar:
        anim.save(filename=str(path), writer="ffmpeg", fps=VIDEO_FPS,
                  progress_callback=lambda i, n: pbar.update(1))
    plt.close(fig)


def save_rollout_video(path, rollout_frames):
    print(f"Saving rollout video to {path}...")
    n = len(rollout_frames)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    title = ax.set_title("Rollout — frame 0")
    img = ax.imshow(rollout_frames[0])

    def update(i):
        title.set_text(f"Rollout — frame {i} / {n - 1}")
        img.set_data(rollout_frames[i])
        return (img,)

    anim = FuncAnimation(fig=fig, func=update, frames=np.arange(n), interval=200)
    with tqdm(total=n) as pbar:
        anim.save(filename=str(path), writer="ffmpeg", fps=VIDEO_FPS,
                  progress_callback=lambda i, _n: pbar.update(1))
    plt.close(fig)


def save_sidebyside_video(
    path,
    rollout_frames,
    px_axis,
    py_axis,
    frames_values,
    ego_xy,
    others_xy_per_frame,
    n_frames,
    vbar=VALUE_CMAP_LIMIT,
    title_suffix_per_frame=None,
):
    print(f"Saving side-by-side video to {path}...")
    n = min(len(rollout_frames), n_frames)
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

    ax_left.set_axis_off()
    left_title = ax_left.set_title("Rollout — frame 0")
    img = ax_left.imshow(rollout_frames[0])

    decorate_brt_axes(ax_right)
    right_title = ax_right.set_title("BRT slice — frame 0")
    mesh = ax_right.pcolormesh(
        px_axis, py_axis, frames_values[0].T,
        cmap="RdBu", vmin=-vbar, vmax=+vbar, shading="auto",
    )
    plt.colorbar(mesh, ax=ax_right, label="V(x, t=1)")
    contour_state = {"artist": ax_right.contour(px_axis, py_axis, frames_values[0].T, levels=[0], colors="k")}
    ego_dot, = ax_right.plot([ego_xy[0][0]], [ego_xy[0][1]], "o", color="lime",
                             markersize=10, markeredgecolor="black", label="ego")
    others0 = np.array(others_xy_per_frame[0]) if others_xy_per_frame[0] else np.zeros((0, 2))
    others_scatter = ax_right.scatter(
        others0[:, 0] if len(others0) else [],
        others0[:, 1] if len(others0) else [],
        s=80, c="blue", edgecolors="black", label="other",
    )
    ax_right.legend(loc="upper right")

    def update(i):
        left_title.set_text(f"Rollout — frame {i} / {n - 1}")
        img.set_data(rollout_frames[i])
        suffix = f" — {title_suffix_per_frame[i]}" if title_suffix_per_frame else ""
        right_title.set_text(f"BRT slice — frame {i} / {n - 1}{suffix}")
        mesh.set_array(frames_values[i].T.ravel())
        contour_state["artist"].remove()
        contour_state["artist"] = ax_right.contour(px_axis, py_axis, frames_values[i].T, levels=[0], colors="k")
        ego_dot.set_data([ego_xy[i][0]], [ego_xy[i][1]])
        pts = np.array(others_xy_per_frame[i]) if others_xy_per_frame[i] else np.zeros((0, 2))
        others_scatter.set_offsets(pts)
        return img, mesh, ego_dot, others_scatter

    anim = FuncAnimation(fig=fig, func=update, frames=np.arange(n), interval=200)
    with tqdm(total=n) as pbar:
        anim.save(filename=str(path), writer="ffmpeg", fps=VIDEO_FPS,
                  progress_callback=lambda i, _n: pbar.update(1))
    plt.close(fig)
