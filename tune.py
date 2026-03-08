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
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src import (
    ScenarioParams,
    GaussianMixtureParam,
    NormalParam,
    ProbabilityParam,
    BetaParam,
    SimulatedEnv,
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
    
    # Learning parameters (narrower range to avoid instability)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-3, log=True)
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


class ProgressCallback(BaseCallback):
    """Callback to print training progress."""
    
    def __init__(self, trial_num: int, total_timesteps: int, print_freq: int = 10000):
        super().__init__()
        self.trial_num = trial_num
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.print_freq == 0:
            pct = 100 * self.num_timesteps / self.total_timesteps
            print(f"[Trial {self.trial_num}] {self.num_timesteps}/{self.total_timesteps} ({pct:.0f}%)")
        return True


class TrialEvalCallback(EvalCallback):
    """Callback for pruning unpromising trials."""
    
    def __init__(self, trial: optuna.Trial, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False
    
    def _on_step(self) -> bool:
        # Check if evaluation happened this step
        old_best = self.best_mean_reward
        result = super()._on_step()
        
        # If best reward changed, an evaluation occurred
        if self.best_mean_reward != old_best:
            print(f"  [Eval {self.eval_idx}] Mean reward: {self.last_mean_reward:.2f}, Best: {self.best_mean_reward:.2f}")
            self.trial.report(self.last_mean_reward, self.eval_idx)
            self.eval_idx += 1
            
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        
        return result


def objective(
    trial: optuna.Trial,
    timesteps: int,
    n_eval_episodes: int,
    eval_freq: int,
    device: str,
) -> float:
    """Optuna objective function."""
    
    print(f"\n[Trial {trial.number}] Starting...")
    
    setup = create_scenario_params()
    
    # Create environments
    print(f"[Trial {trial.number}] Creating environments...")
    env = Monitor(gym.make("SimulatedEnv-v0", render_mode=None, scenario_params=setup))
    eval_env = Monitor(gym.make("SimulatedEnv-v0", render_mode=None, scenario_params=setup))
    print(f"[Trial {trial.number}] Environments created.")
    
    # Sample hyperparameters
    params = sample_dqn_params(trial)
    print(f"[Trial {trial.number}] Params: lr={params['learning_rate']:.6f}, batch={params['batch_size']}")
    
    # Create model
    print(f"[Trial {trial.number}] Creating model...")
    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        device=device,
        **params,
    )
    print(f"[Trial {trial.number}] Model created.")
    
    # Callbacks
    progress_callback = ProgressCallback(trial.number, timesteps, print_freq=10000)
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
        print(f"[Trial {trial.number}] Starting training for {timesteps} timesteps...")
        model.learn(total_timesteps=timesteps, callback=[progress_callback, eval_callback], progress_bar=False)
        
        # Get the reward - use last_mean_reward if best is not set
        reward = eval_callback.best_mean_reward
        if reward == float("-inf") and hasattr(eval_callback, 'last_mean_reward'):
            reward = eval_callback.last_mean_reward
        
        # Check for invalid reward
        if reward is None or np.isnan(reward) or reward == float("-inf"):
            print(f"[Trial {trial.number}] Invalid reward ({reward}), marking as failed.")
            nan_encountered = True
        else:
            print(f"[Trial {trial.number}] Training complete. Best reward: {reward:.2f}")
    except Exception as e:
        print(f"[Trial {trial.number}] Failed: {e}")
        nan_encountered = True
        reward = None
    finally:
        env.close()
        eval_env.close()
    
    if nan_encountered:
        return float("-inf")
    
    if eval_callback.is_pruned:
        print(f"[Trial {trial.number}] Pruned.")
        raise optuna.exceptions.TrialPruned()
    
    return reward


def tune(
    n_trials: int = 50,
    timesteps: int = 100_000,
    n_eval_episodes: int = 5,
    eval_freq: int = 5000,
    device: str = "auto",
    study_name: str = "dqn_roundabout",
    storage: Optional[str] = None,
    n_jobs: int = 1,
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
    if n_jobs != 1:
        print(f"Parallel jobs: {n_jobs if n_jobs > 0 else 'all cores'}")
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
            n_jobs=n_jobs,
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
    parser.add_argument(
        "--n-jobs", "-j",
        type=int,
        default=1,
        help="Number of parallel trials (use -1 for all CPU cores)"
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
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
