#!/usr/bin/env python3
"""
Hyperparameter tuning script using Optuna for DQN on SimulatedEnv.

Usage:
    python tune.py                      # Run 50 trials (default)
    python tune.py --n-trials 100       # Run 100 trials
    python tune.py --n-trials 20 --timesteps 50000  # Quick test

Install Optuna first:
    pip install optuna optuna-dashboard
"""

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

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
        initial_position_x=GaussianMixtureParam(p=[1.0], mu=[0.0, 0.0], sigma=[0.005, 0.005]),
        initial_position_y=GaussianMixtureParam(p=[1.0], mu=[0.0, 0.0], sigma=[0.005, 0.005]),
        velocity_x=GaussianMixtureParam(p=[1.0], mu=[0.0, 0.0], sigma=[SQRT_2 * 0.005 / 0.1, SQRT_2 * 0.005 / 0.1]),
        velocity_y=GaussianMixtureParam(p=[1.0], mu=[0.0, 0.0], sigma=[SQRT_2 * 0.005 / 0.1, SQRT_2 * 0.005 / 0.1]),
        high_lvl_ctrl_noise=ProbabilityParam(p=[0.0]),
        initial_speed=NormalParam(mu=8.0, sigma=0.5),
        initial_heading=ProbabilityParam(p=[0.33, 0.33, 0.33]),
        politeness=BetaParam(ab=[1, 1]),
        other_vehicle_speed=NormalParam(mu=16.0, sigma=2.0),
        entering_vehicle_position=GaussianMixtureParam(p=[1.0], mu=[5.0, 5.0], sigma=[2.0, 2.0])
    )


def sample_dqn_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample hyperparameters for DQN."""
    
    # Network architecture
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "large"])
    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "large": [512, 256, 128],
    }[net_arch_type]
    
    # Learning parameters
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.7, 0.999)
    
    # Buffer and exploration
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000, 200000])
    learning_starts = trial.suggest_int("learning_starts", 100, 10000)
    
    # Target network update
    target_update_interval = trial.suggest_int("target_update_interval", 1, 1000)
    
    # Training frequency
    train_freq = trial.suggest_int("train_freq", 1, 16)
    gradient_steps = trial.suggest_int("gradient_steps", 1, 8)
    
    # Exploration
    exploration_fraction = trial.suggest_float("exploration_fraction", 0.05, 0.5)
    exploration_final_eps = trial.suggest_float("exploration_final_eps", 0.01, 0.2)
    
    return {
        "policy_kwargs": dict(net_arch=net_arch),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "gamma": gamma,
        "buffer_size": buffer_size,
        "learning_starts": learning_starts,
        "target_update_interval": target_update_interval,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "exploration_fraction": exploration_fraction,
        "exploration_final_eps": exploration_final_eps,
    }


class TrialEvalCallback(EvalCallback):
    """Callback for pruning unpromising trials."""
    
    def __init__(self, trial: optuna.Trial, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
    
    def _on_step(self) -> bool:
        result = super()._on_step()
        
        if self.eval_idx > 0:
            self.trial.report(self.last_mean_reward, self.eval_idx)
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        
        self.eval_idx += 1
        return result


def objective(
    trial: optuna.Trial,
    timesteps: int,
    n_eval_episodes: int,
    eval_freq: int,
    device: str,
) -> float:
    """Optuna objective function."""
    
    setup = create_scenario_params()
    
    # Create environments
    env = Monitor(gym.make("SimulatedEnv-v0", render_mode=None, scenario_params=setup))
    eval_env = Monitor(gym.make("SimulatedEnv-v0", render_mode=None, scenario_params=setup))
    
    # Sample hyperparameters
    params = sample_dqn_params(trial)
    
    # Create model
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        device=device,
        **params,
    )
    
    # Evaluation callback with pruning
    eval_callback = TrialEvalCallback(
        trial,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        deterministic=True,
        verbose=0,
    )
    
    nan_encountered = False
    try:
        model.learn(total_timesteps=timesteps, callback=eval_callback)
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        nan_encountered = True
    finally:
        env.close()
        eval_env.close()
    
    if nan_encountered:
        return float("-inf")
    
    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()
    
    return eval_callback.best_mean_reward


def tune(
    n_trials: int = 50,
    timesteps: int = 100_000,
    n_eval_episodes: int = 5,
    eval_freq: int = 5000,
    device: str = "auto",
    study_name: str = "dqn_roundabout",
    storage: Optional[str] = None,
):
    """Run hyperparameter tuning."""
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"Device: {device}")
    print(f"Running {n_trials} trials, {timesteps:,} timesteps each")
    print()
    
    # Create Optuna study
    sampler = TPESampler(n_startup_trials=10, seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )
    
    try:
        study.optimize(
            lambda trial: objective(trial, timesteps, n_eval_episodes, eval_freq, device),
            n_trials=n_trials,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTuning interrupted.")
    
    # Print results
    print("\n" + "=" * 60)
    print("TUNING RESULTS")
    print("=" * 60)
    
    print(f"\nNumber of finished trials: {len(study.trials)}")
    
    print(f"\nBest trial (#{study.best_trial.number}):")
    print(f"  Mean reward: {study.best_trial.value:.2f}")
    print(f"\n  Hyperparameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best params to file
    save_path = Path("roundabout_dqn")
    save_path.mkdir(exist_ok=True)
    
    with open(save_path / "best_params.txt", "w") as f:
        f.write(f"Best trial: #{study.best_trial.number}\n")
        f.write(f"Mean reward: {study.best_trial.value:.2f}\n\n")
        f.write("Hyperparameters:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"\nBest parameters saved to {save_path / 'best_params.txt'}")
    
    # Print command to train with best params
    print("\nTo train with best parameters, run:")
    params = study.best_trial.params
    cmd = f"python train.py --timesteps 500000"
    if "learning_rate" in params:
        cmd += f" --learning-rate {params['learning_rate']:.6f}"
    if "batch_size" in params:
        cmd += f" --batch-size {params['batch_size']}"
    if "gamma" in params:
        cmd += f" --gamma {params['gamma']:.4f}"
    if "buffer_size" in params:
        cmd += f" --buffer-size {params['buffer_size']}"
    print(f"  {cmd}")
    
    return study


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for DQN using Optuna",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-trials", "-n",
        type=int,
        default=50,
        help="Number of trials to run"
    )
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=100_000,
        help="Timesteps per trial"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="Episodes per evaluation"
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="Evaluation frequency"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to use"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="dqn_roundabout",
        help="Optuna study name"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///optuna.db)"
    )
    
    args = parser.parse_args()
    
    tune(
        n_trials=args.n_trials,
        timesteps=args.timesteps,
        n_eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        device=args.device,
        study_name=args.study_name,
        storage=args.storage,
    )


if __name__ == "__main__":
    main()
