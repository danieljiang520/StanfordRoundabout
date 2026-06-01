# CLAUDE.md

Working notes for Claude Code on this repository. Documents what each file is for so future sessions can navigate without re-exploring.

## Project overview

Research code for a Stanford roundabout autonomous-driving project. Combines three things:

1. A DQN-GNN policy trained on `highway-env`'s roundabout (under `roundabout_dqn_gnn/`).
2. A DeepReach two-car BRT (Backward Reachable Tube) value function, trained for 30k epochs and stored as `vf_30k_epoch.ckpt`.
3. Visualization, evaluation, and a BRT-triggered safety controller that wraps the nominal DQN-GNN policy and stops the ego when its value drops below a margin. See `README_BRT_SAFETY_CONTROLLER.md` for the full controller writeup.

Conda env: `highway` (Python 3.10). Activate with `conda activate highway`.

## Top-level layout

```
StanfordRoundabout/
  README.md                       conda setup
  README_BRT_SAFETY_CONTROLLER.md detailed notes on the safety controller and findings
  CLAUDE.md                       this file
  train_TwoCar.py                 launches DeepReach training for the original TwoCar8D BRT
  train_TwoCar_body_geometry.py   launches training for a body-geometry variant of TwoCar8D
  vf_30k_epoch.ckpt               trained BRT value-network checkpoint (30k epochs)
  test_roundabout.py              standalone smoke test for the roundabout env
  pyproject.toml, requirements.txt
  scripts/                        rollouts, visualization, evaluation, baseline RL training
  src/                            project package (importable as `src`)
  libraries/DeepReach_MPC/        vendored DeepReach codebase (dynamics, modules, training)
  homework/AA276_GCP/             course homework reference (multi-obstacle pattern, NeuralVF)
  roundabout_dqn_gnn/             trained policy and outputs (model.zip, videos/, safety_eval/)
  roundabout_dqn/                 older DQN baseline runs
  notebook/                       Jupyter notebooks (analysis, baselines, fuzzing experiments)
  other/                          standalone helper scripts (train.py, tune.py)
  figures/                        plots produced by the notebooks
```

## scripts/

Entry points for training, evaluating, and visualizing policies.

### BRT visualization and safety

- `scripts/brt_utils.py`. Shared helpers for everything BRT-related. Loads `vf_30k_epoch.ckpt` into a `SingleBVPNet`, monkey-patches `TwoCar8D`'s hard-coded `.cuda()` calls so inference runs on CPU, and provides:
  - `build_cpu_dynamics()`, `load_value_network(dynamics)`.
  - `make_roundabout_config(traffic_vehicles)`, `make_grid()`.
  - `query_slice_single(...)`, `query_slice_multi(...)`, `values_at_ego_state(...)`.
  - `save_brt_video(...)`, `save_rollout_video(...)`, `save_sidebyside_video(...)`.
  - Constants: `COLLISION_R = 1.0`, `WHEELBASE = 5.0`, `T_QUERY = 1.0`, `GRID_EXTENT = 60.0`, `GRID_RESOLUTION = 100`, `VALUE_CMAP_LIMIT = 50.0`, `VIDEO_FPS = 5`, `EPISODE_DURATION = 20`.
  - The previous version of this file is kept commented out above the active code for reference.

- `scripts/brt_utils_body_geometry.py`. Sister helper module for the body-geometry variant of `TwoCar8D` (trained via `train_TwoCar_body_geometry.py`). Not all visualization scripts use it.

- `scripts/visualize_brt_rollout.py`. Single-other-vehicle rollout video. `TRAFFIC_VEHICLES = 1`. Tries 20 seeds, picks the longest non-crashing rollout, renders the rollout video, the BRT slice video, and a side-by-side video to `roundabout_dqn_gnn/videos/`.

- `scripts/visualize_brt_rollout_multicar.py`. Multi-vehicle (`TRAFFIC_VEHICLES = 3`) variant. Each frame, evaluates V once per other vehicle at the ego's actual state, picks the most-dangerous vehicle (the one with the lowest V), plots only that vehicle's BRT slice, and draws a blue dot for every traffic vehicle. Only saves the side-by-side video.

- `scripts/visualize_brt_safety_rollouts_multicar.py`. Records paired nominal and BRT-safety-controlled videos for fixed seeds. The safety controller monitors the most-dangerous vehicle's V at every policy step and overrides the DQN action with a hard stop when `V_ego < SAFETY_VALUE_THRESHOLD`. Saves `nominal_rollout_seed_NN.mp4` and `safety_rollout_and_brt_seed_NN.mp4` files.

- `scripts/visualize_brt_four_action_safety_rollouts_multicar.py`. Four-action safety-controller visualization. Keeps the nominal DQN-GNN policy, but when the BRT filter intervenes it chooses among `FORWARD`, `STOP`, `LEFT`, and `RIGHT` by one-step BRT value prediction on deep-copied envs. `FORWARD` maps to highway-env `FASTER`; `LEFT`/`RIGHT` map to lane-change meta-actions; `STOP` preserves the hard stop override (`speed = 0`, `target_speed = 0`, then `IDLE`). Supports `--filter least-restrictive` (`V < 0` intervention) and `--filter smooth --gamma <value>` (finite-difference `Vdot + gamma V >= 0` smooth blending). Saves `four_action_least_restrictive_seed_NN.mp4` or `four_action_smooth_gamma_G_seed_NN.mp4`.

- `scripts/evaluate_brt_safety_rate.py`. Non-rendered paired evaluation: runs many seeds with and without the BRT safety wrapper, sweeps over `traffic_vehicles_count`, and writes per-rollout CSVs and a summary JSON to `roundabout_dqn_gnn/safety_eval/<timestamp>_traffic_NN/`.

### Policy training and baselines

- `scripts/sb3_roundabout_dqn_gnn.py`. The main DQN-GNN training script. Defines `RoundaboutGNNExtractor` (masked graph extractor over the padded Kinematics observation) and trains a stable-baselines3 DQN against `roundabout-v0`. Output goes to `roundabout_dqn_gnn/`.
- `scripts/sb3_roundabout_dqn.py`. Baseline DQN (no GNN) on the roundabout.
- `scripts/sb3_highway_dqn.py`, `sb3_highway_ppo.py`, `sb3_highway_dqn_cnn.py`, `sb3_highway_ppo_transformer.py`. Upstream highway-env examples kept verbatim.
- `scripts/sb3_racetrack_oval_ppo.py`, `sb3_racetracks_ppo.py`. Upstream racetrack examples.
- `scripts/utils.py`. Small helpers used by the sb3 scripts.

## src/

Importable package. `src/__init__.py` registers `VariableRoundabout-v0` and re-exports the public API listed below.

- `src/scenario_params.py`. Dataclasses for fuzzable scenario parameters: `NormalParam`, `BetaParam`, `GaussianMixtureParam`, `ProbabilityParam`, `ScenarioParams`. Defines `NOMINAL` (the baseline configuration) and `SQRT_2`.
- `src/distributions.py`. PDF and sampling helpers (`to_tensor`, `sample_gaussian_mixture`, `gaussian_pdf`, `gaussian_mixture_pdf`).
- `src/vehicle.py`. `CustomPoliteVehicle`, an `IDMVehicle` with a learnable politeness factor.
- `src/simulated_env.py`. `SimulatedEnv`, a roundabout environment whose scenario parameters are wired to `ScenarioParams`.
- `src/variable_roundabout_env.py`. `VariableRoundaboutEnv`. RoundaboutEnv variant whose traffic spawn count is controlled by `traffic_vehicles_count`. Registered as `VariableRoundabout-v0`.
- `src/fuzzer.py`. `ScenarioFuzzer` and `FuzzerConfig`. Optimizes scenario parameters to find failure cases.
- `src/robustness.py`. `compute_robustness`, `trajectory_metrics_from_rollout`, `weights_from_vector`. Computes a weighted robustness score over safety, stability, efficiency, road-keeping, and hard-braking sub-scores.
- `src/robustness_helpers.py`, `src/robustness_optimize.py`. Sampling utilities and weight optimization for robustness scoring.
- `src/failure_probability.py`. `estimate_failure_probability`, `importance_sampling_estimate`, `print_failure_probability_report`. Bayesian and importance-sampling failure-rate estimators.
- `src/rollout_states.py`. `VehicleState` dataclass, `extract_vehicle_states(env)`, `rollout_with_states(env, model, ...)`. Walks `env.unwrapped.road.vehicles` to return per-timestep `(px, py, psi, v)` for ego and every other vehicle, with optional rgb_array frame capture. Used by every BRT visualization and evaluation script.
- `src/plotting/`. Plot helpers used by the notebooks.

## libraries/DeepReach_MPC/

Vendored copy of the DeepReach codebase. Relevant pieces:

- `libraries/DeepReach_MPC/dynamics/dynamics.py`. All the Dynamics subclasses. `TwoCar8D` is the one used here. Note that this file currently contains two `TwoCar8D` definitions (original around line 1508 and a body-geometry variant around line 1792). State layout is `[px1, py1, psi1, v1, px2, py2, psi2, v2]`, `state_range_` is `[-100, 100]` on positions and `[-pi, pi]` on headings, `v` is `[0, 32]` m/s. `boundary_fn` is `(px1-px2)^2 + (py1-py2)^2 - collisionR^2`. The class hard-codes `.cuda()` on its tensor buffers, which is why `brt_utils.build_cpu_dynamics()` patches it for CPU inference.
- `libraries/DeepReach_MPC/utils/modules.py`. `SingleBVPNet` is the network class loaded from `vf_30k_epoch.ckpt`. Hidden width 512, depth 3, sine activations, input dim 11 (time plus a periodic-transformed state).
- `libraries/DeepReach_MPC/run_experiment.py`. The training entry point that `train_TwoCar.py` and `train_TwoCar_body_geometry.py` shell out to.

## homework/AA276_GCP/

Reference code from a course. Two pieces are reused conceptually by this project:

- `homework/AA276_GCP/hw3/problem3_helper.py`. Contains `NeuralVF`, the canonical pattern for loading a `SingleBVPNet` checkpoint and querying `.values(x)` and `.gradients(x)`. `brt_utils.load_value_network` follows the same shape.
- `homework/AA276_GCP/hw3/problem3.py`. `smooth_blending_safety_filter` selects the most-threatening obstacle via `Vs.argmin()` (after shifting V per obstacle). The multi-vehicle BRT scripts use the same "min over obstacles" pattern, but without coordinate shift because the TwoCar8D BRT already accepts the full state of both cars as input.
- `homework/AA276_GCP/hw3/problem1_helper.py`. `save_values_gif`. Original template for the matplotlib-based BRT animation now in `brt_utils.save_brt_video`.

## roundabout_dqn_gnn/

Outputs of the GNN policy.

- `roundabout_dqn_gnn/model.zip`. Trained stable-baselines3 DQN-GNN policy.
- `roundabout_dqn_gnn/DQN_1/`, `DQN_2/`. TensorBoard logs.
- `roundabout_dqn_gnn/videos/`. MP4s produced by the visualization scripts: `brt_overlay.mp4`, `rollout.mp4`, `rollout_and_brt.mp4` (single-other-car); `rollout_and_brt_multicar.mp4` (multi-car); `nominal_rollout_seed_NN.mp4`, `safety_rollout_and_brt_seed_NN.mp4` (stop safety controller); `four_action_least_restrictive_seed_NN.mp4`, `four_action_smooth_gamma_G_seed_NN.mp4` (four-action filters).
- `roundabout_dqn_gnn/safety_eval/`. CSV and JSON outputs from `evaluate_brt_safety_rate.py`. Each subdirectory is one run, named by timestamp (and optionally `_traffic_NN` when sweeping over `traffic_vehicles_count`).

## notebook/

- `notebook/project.ipynb`. Main project notebook.
- `notebook/disturbances-testing.ipynb`, `notebook/traj_distribution.ipynb`. Fuzzer and distribution analysis.
- `notebook/sensitivity_analysis.ipynb`, `notebook/timestep_sensitivity_analysis.ipynb`. Sensitivity studies (One-at-a-Time, Shapley, etc., with outputs in `figures/`).
- The other notebooks (highway_planning, intersection_social_dqn, parking_her, parking_model_based, sb3_highway_dqn) are upstream highway-env examples.

## Common workflows

### Run a single-car BRT visualization

```bash
python scripts/visualize_brt_rollout.py
```

Produces three MP4s in `roundabout_dqn_gnn/videos/`.

### Run a multi-car BRT visualization

```bash
python scripts/visualize_brt_rollout_multicar.py
```

Produces `rollout_and_brt_multicar.mp4` only. Edit `TRAFFIC_VEHICLES` at the top of the script to change the number of traffic vehicles.

### Generate the BRT safety controller demo

```bash
python scripts/visualize_brt_safety_rollouts_multicar.py
```

Produces paired nominal and safety MP4s for each seed.

### Generate the four-action BRT safety controller demo

```bash
python scripts/visualize_brt_four_action_safety_rollouts_multicar.py --filter least-restrictive
python scripts/visualize_brt_four_action_safety_rollouts_multicar.py --filter smooth --gamma 5
```

Produces paired nominal and four-action safety MP4s for each seed. Use `--num-seeds 1` for a quick smoke run.

### Evaluate the safety controller across many seeds

```bash
python scripts/evaluate_brt_safety_rate.py
```

Writes CSV/JSON to a timestamped `roundabout_dqn_gnn/safety_eval/<run>/` directory.

### Retrain the BRT

```bash
python train_TwoCar.py                  # original TwoCar8D
python train_TwoCar_body_geometry.py    # body-geometry variant
```

Both shell out to `libraries/DeepReach_MPC/run_experiment.py`.

### Retrain the DQN-GNN policy

Set `TRAIN = True` in `scripts/sb3_roundabout_dqn_gnn.py`, then run it.

## Notes for future Claude sessions

- The TwoCar8D BRT was trained with `collisionR=1.0`, `wheelbase=5.0`, `set_mode='avoid'` (see the active config in `train_TwoCar.py`). Those values must match when loading the checkpoint.
- `COLLISION_R = 1.0` is a center-to-center distance in the value function's frame, not a body-to-body distance on the rendered vehicles.
- highway-env's world frame and TwoCar8D's frame agree directly: `vehicle.position`, `vehicle.heading`, `vehicle.speed` map to `(px, py, psi, v)` with no transformation. Both integrate the same bicycle model.
- highway-env's y-axis points down (screen convention). The BRT plots invert the y-axis in `bu.decorate_brt_axes` so the orientation matches the rollout render.
- The simple `V < 0` trigger is too late under discrete control; the current scripts use a positive margin (`SAFETY_VALUE_THRESHOLD = 50.0`).
- `policy_frequency` must be raised above the highway-env default (1 Hz) for the safety controller to react in time. Current scripts use 15 Hz.
- In the four-action safety script, smooth blending is implemented in discrete time as `(V_next - V_current) / dt + gamma * V_current >= 0`, because the deployed controller selects highway-env meta-actions instead of direct continuous controls.
- The `dynamics.py` file currently contains two `TwoCar8D` classes (original and body-geometry). Be careful which one Python imports if you touch the file.
