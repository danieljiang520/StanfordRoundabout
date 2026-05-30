# BRT Safety Controller Rollout Notes

This note documents the DQN-GNN roundabout rollout visualizers, the BRT-based
safety controller, and the main debugging conclusions from the development
conversation.

## Main Scripts

- `scripts/visualize_brt_rollout_multicar.py`
  - Visualizes a nominal DQN-GNN rollout with BRT overlays.
  - Uses 3 traffic vehicles.
  - The BRT model is a two-car value function, so each frame evaluates the ego
    state against every traffic vehicle and selects the most dangerous one:
    the vehicle with the lowest value at the current ego state.
  - Only that selected vehicle's BRT slice is shown in the heatmap.

- `scripts/visualize_brt_safety_rollouts_multicar.py`
  - Runs fixed-seed rollouts with both nominal DQN and BRT safety control.
  - Saves nominal-only videos and safety/BRT side-by-side videos.
  - Uses the same most-dangerous-vehicle display convention as
    `visualize_brt_rollout_multicar.py`: all traffic cars are shown as points,
    but only one BRT field is rendered.

- `scripts/brt_utils.py`
  - Provides shared helpers for loading the DeepReach value network, creating
    the roundabout config, evaluating BRT grids, evaluating values at the ego
    state, and exporting videos.

## Safety Controller Logic

The safety script keeps the trained DQN-GNN as the nominal controller. At every
policy step:

1. Extract ego and traffic-vehicle states with `src.extract_vehicle_states`.
2. Evaluate the TwoCar8D value function once per traffic vehicle at the ego's
   current state.
3. Select the most dangerous traffic vehicle:

   ```python
   dangerous_idx = int(np.argmin(v_at_ego))
   ```

4. Render only that vehicle's BRT slice with `bu.query_slice_single`.
5. Use the selected minimum value as the ego safety value.
6. If the value is below the configured threshold, override the DQN action with
   the stop controller.

The stop controller is intentionally simple:

```python
controlled_ego.speed = 0.0
controlled_ego.target_speed = 0.0
action = IDLE
```

When the controller first switches to `STOP`, the ego vehicle's previous
`target_speed` and `speed_index` are saved. When the value recovers and the
controller returns to `DQN`, those saved nominal controller settings are
restored so the temporary safety stop does not permanently suppress motion.

## Current Safety Parameters

The current safety rollout script uses:

```python
TRAFFIC_VEHICLES = 3
ROLLOUT_SEEDS = range(10)
SIMULATION_FREQUENCY = 15
POLICY_FREQUENCY = 5
SAFETY_VALUE_THRESHOLD = 50.0
```

The safety test is:

```python
mode = "STOP" if ego_value < SAFETY_VALUE_THRESHOLD else "DQN"
```

This is more conservative than the raw BRT boundary test `V < 0`. The positive
margin gives the stop action time to take effect under discrete policy updates.

## Video Outputs

Videos are saved under:

```text
roundabout_dqn_gnn/videos/
```

Nominal-only videos:

```text
nominal_rollout_seed_00.mp4
...
nominal_rollout_seed_09.mp4
```

Safety-controlled side-by-side videos:

```text
safety_rollout_and_brt_seed_00.mp4
...
safety_rollout_and_brt_seed_09.mp4
```

Each side-by-side video shows:

- Left: rendered roundabout rollout.
- Right: BRT slice for the currently most dangerous traffic vehicle.
- Green dot: ego vehicle.
- Blue dots: traffic vehicles.
- Title: current frame, ego value, controller mode, and selected minimum-value
  traffic vehicle index.

## Important Findings

### `COLLISION_R = 1.0`

The BRT collision radius is interpreted as a center-to-center distance in the
value-function dynamics, not as a boundary-to-boundary vehicle-body distance.
That means `COLLISION_R = 1.0` is not the same thing as saying the rendered car
rectangles collide only when their visual boundaries are 1 meter apart.

### Why `V < 0` Was Not Enough

Using a strict `V < 0` trigger can be too late in this simulator because control
updates are discrete and the stop action is applied only at policy decision
times. A car can enter the dangerous set between checks or be too close for the
simple stop override to prevent the simulator-level collision.

### Policy Frequency Matters

The default policy update rate sampled safety too slowly for some seeds. Raising
`policy_frequency` makes the controller check BRT more often. In the current
script:

```python
SIMULATION_FREQUENCY = 15
POLICY_FREQUENCY = 5
```

This checks safety 5 times per second while the environment dynamics integrate
at 15 Hz.

Higher rates such as 10 Hz or 15 Hz can reduce delay, but frequency alone did
not solve every failing case when the threshold was still `V < 0`.

### Positive Threshold/Margin

The successful fix was to intervene before the raw BRT boundary:

```python
SAFETY_VALUE_THRESHOLD = 50.0
```

This means `STOP` begins when `V_ego < 50`, not only when `V_ego < 0`.

In the experiments discussed:

- Thresholds `0`, `10`, and `25` were not enough for seed 07 at 5 Hz.
- Threshold `50` avoided the seed 07 collision.
- With threshold `50`, all fixed seeds `0` through `9` avoided collision in the
  tested safety replay.

### Nominal Rollouts Show the Baseline Failure

Nominal DQN-only videos are saved separately to show what happens without the
BRT safety controller. In the tested nominal rollouts at the current 5 Hz
setting, the crashing seeds were:

```text
03, 05, 07, 08
```

These nominal clips are useful presentation artifacts because they show that
the safety controller changes the outcome rather than merely visualizing the
same behavior.

### Why BRT Can Look Dangerous When Cars Look Far Apart

The BRT value is not just Euclidean distance on the rendered frame. It depends
on the full two-car state:

- ego position,
- ego heading,
- ego speed,
- other vehicle position,
- other vehicle heading,
- other vehicle speed,
- the learned time-horizon value function.

So a visually separated configuration can still have a low value if the
relative heading and speed make future collision reachable.

## Reproducing

Compile-check the safety script:

```bash
conda run -n highway python -m py_compile scripts/visualize_brt_safety_rollouts_multicar.py
```

Generate all nominal and safety videos:

```bash
conda run -n highway python scripts/visualize_brt_safety_rollouts_multicar.py
```

This may take several minutes because every safety frame evaluates the BRT
network over a grid and then encodes MP4 video.

## Current Presentation Story

The clean story is:

1. Run the nominal DQN-GNN policy and show that some seeds collide.
2. Run the same seeds with the BRT safety wrapper.
3. The safety wrapper monitors the most dangerous traffic vehicle at each
   decision step.
4. When `V_ego < 50`, it temporarily stops the ego vehicle.
5. When the value recovers, it restores the nominal controller state and returns
   to DQN.
6. The videos show that the BRT safety intervention avoids collisions that the
   nominal controller alone does not.
