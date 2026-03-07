#!/usr/bin/env python3
"""
Training script for DQN model on the SimulatedEnv roundabout environment.

Usage:
    python train.py                          # Train with defaults (100k timesteps)
    python train.py --timesteps 500000       # Train for 500k timesteps
    python train.py --resume                 # Resume training from existing model
    python train.py --timesteps 200000 --resume  # Resume and train for 200k more
"""

import argparse
import os
from pathlib import Path

import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from src import (
    ScenarioParams,
    GaussianMixtureParam,
    NormalParam,
    ProbabilityParam,
    BetaParam,
    SQRT_2,
)


def create_scenario_params() -> ScenarioParams:
    """Create the scenario parameters for training."""
    return ScenarioParams(
        # Observation disturbances
        initial_position_x=GaussianMixtureParam(
            p=[1.0],
            mu=[0.0, 0.0],
            sigma=[0.005, 0.005]
        ),
        initial_position_y=GaussianMixtureParam(
            p=[1.0],
            mu=[0.0, 0.0],
            sigma=[0.005, 0.005]
        ),
        velocity_x=GaussianMixtureParam(
            p=[1.0],
            mu=[0.0, 0.0],
            sigma=[SQRT_2 * 0.005 / 0.1, SQRT_2 * 0.005 / 0.1]
        ),
        velocity_y=GaussianMixtureParam(
            p=[1.0],
            mu=[0.0, 0.0],
            sigma=[SQRT_2 * 0.005 / 0.1, SQRT_2 * 0.005 / 0.1]
        ),
        # Action disturbances
        high_lvl_ctrl_noise=ProbabilityParam(p=[0.0]),
        initial_speed=NormalParam(mu=8.0, sigma=0.5),
        initial_heading=ProbabilityParam(p=[0.33, 0.33, 0.33]),
        # Environment disturbances
        politeness=BetaParam(ab=[1, 1]),
        other_vehicle_speed=NormalParam(mu=16.0, sigma=2.0),
        entering_vehicle_position=GaussianMixtureParam(
            p=[1.0],
            mu=[5.0, 5.0],
            sigma=[2.0, 2.0]
        )
    )


def train(
    timesteps: int = 100_000,
    resume: bool = False,
    save_dir: str = "roundabout_dqn",
    checkpoint_freq: int = 10_000,
    eval_freq: int = 5_000,
    learning_rate: float = 5e-4,
    buffer_size: int = 15_000,
    batch_size: int = 32,
    gamma: float = 0.8,
    device: str = "auto",
):
    """
    Train the DQN model.

    Args:
        timesteps: Total timesteps to train for
        resume: If True, load existing model and continue training
        save_dir: Directory to save model and checkpoints
        checkpoint_freq: Save checkpoint every N timesteps
        eval_freq: Evaluate model every N timesteps
        learning_rate: Learning rate for optimizer
        buffer_size: Size of replay buffer
        batch_size: Batch size for training
        gamma: Discount factor
        device: Device to use ("auto", "cuda", "cpu", or "mps" for Apple Silicon)
    """
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    model_path = save_path / "model.zip"

    # Create scenario params
    setup = create_scenario_params()

    # Create training environment
    env = gym.make("SimulatedEnv-v0", render_mode=None, scenario_params=setup)

    # Create eval environment (separate instance)
    eval_env = gym.make("SimulatedEnv-v0", render_mode=None, scenario_params=setup)

    # Determine actual device
    if device == "auto":
        if torch.cuda.is_available():
            actual_device = "cuda"
        elif torch.backends.mps.is_available():
            actual_device = "mps"
        else:
            actual_device = "cpu"
    else:
        actual_device = device

    print(f"\nDevice: {actual_device}")
    if actual_device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif actual_device == "mps":
        print("  Using Apple Silicon GPU (MPS)")

    if resume and model_path.exists():
        print(f"\nResuming training from {model_path}")
        model = DQN.load(model_path, env=env, device=actual_device)
        model.learning_rate = learning_rate
    else:
        print("\nCreating new model")
        model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=200,
            batch_size=batch_size,
            gamma=gamma,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            device=actual_device,
        )

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(save_path / "checkpoints"),
        name_prefix="dqn_roundabout",
        save_replay_buffer=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(save_path / "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )

    # Train
    print(f"\nTraining for {timesteps:,} timesteps...")
    print(f"  Device: {actual_device}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Buffer size: {buffer_size:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gamma: {gamma}")
    print(f"  Checkpoints: every {checkpoint_freq:,} timesteps")
    print(f"  Evaluation: every {eval_freq:,} timesteps")
    print()

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    model.save(str(save_path / "model"))
    print(f"\nModel saved to {save_path / 'model.zip'}")

    env.close()
    eval_env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN model on SimulatedEnv roundabout",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=100_000,
        help="Total timesteps to train for"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume training from existing model"
    )
    parser.add_argument(
        "--save-dir", "-s",
        type=str,
        default="roundabout_dqn",
        help="Directory to save model and checkpoints"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="Save checkpoint every N timesteps"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5_000,
        help="Evaluate model every N timesteps"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=5e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=15_000,
        help="Replay buffer size"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="Discount factor"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use (auto detects GPU)"
    )

    args = parser.parse_args()

    train(
        timesteps=args.timesteps,
        resume=args.resume,
        save_dir=args.save_dir,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        device=args.device,
    )


if __name__ == "__main__":
    main()
