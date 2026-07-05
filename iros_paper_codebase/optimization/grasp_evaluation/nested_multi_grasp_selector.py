

#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Nested Multi-Hand Multi-Task Grasp Selector

Tests multiple robot hands across multiple tasks with nested grasp optimization.
Each hand-object combination has its own TPE optimizer for grasp parameters.

The simulation loads Isaac Sim ONCE and runs multiple grasp optimization iterations
without restarting, dramatically improving efficiency.

Key differences from multi_grasp_selector.py:
- Instantiates GraspOptimizer internally for each hand-object combination
- Runs n_grasp_iter iterations of grasp optimization within a single Isaac Sim session
- Samples n_grasp_sample grasp configs per combination per iteration
- Computes final grasp score as average of top_k wrench scores
- Outputs hand scores (avg across all objects) for nested hand optimization

Usage:
    python nested_multi_grasp_selector.py \\
        --urdf_folder urdfs/ \\
        --n_grasp_iter 10 \\
        --n_grasp_sample 32 \\
        --top_k 5 \\
        --device cuda:0 \\
        --output_dir ./nested_selector_results
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher # type: ignore
script_start_time = time.time()
# Parse arguments before launching
parser = argparse.ArgumentParser(description="Nested multi-hand multi-task grasp selector with TPE optimization.")
parser.add_argument("--urdf_folder", type=str, required=True,
                    help="Path to folder containing hand URDF subfolders")
parser.add_argument("--n_grasp_iter", type=int, default=10,
                    help="Number of grasp optimization iterations per hand-object combination")
parser.add_argument("--n_grasp_sample", type=int, default=32,
                    help="Number of grasp samples per hand-object combination per iteration")
parser.add_argument("--top_k", type=int, default=5,
                    help="Number of top grasps to average for grasp score")
parser.add_argument("--device", type=str, default='cuda:0',
                    help="CUDA device to run simulation on")
parser.add_argument("--output_dir", type=str, default='./nested_selector_results',
                    help="Output directory for results")
parser.add_argument("--tasks", type=str, default='hammer,stir,knife',
                    help="Comma-separated list of tasks to test")
parser.add_argument("--livestream", type=int, default=0,
                    help="(0) None, (1) Deprecated native stream, (2) WebRTC stream")
parser.add_argument("--headless", action="store_true", default=True,
                    help="Run in headless mode")
parser.add_argument("--seed", type=int, default=0,
                    help="Random seed for reproducibility (sets torch, numpy, and python random seeds)")
parser.add_argument("--gpu_memory_fraction", type=float, default=1.0,
                    help="Fraction of GPU memory to use (0.0 to 1.0, e.g., 0.5 for 50%%)")
parser.add_argument("--gpu_compute_fraction", type=float, default=1.0,
                    help="Fraction of GPU compute to use (0.0 to 1.0, e.g., 0.5 for 50%%). Limits parallel environments proportionally.")
parser.add_argument("--physx_num_threads", type=int, default=None,
                    help="Number of PhysX CPU threads (None = auto). Lower values reduce CPU/GPU load.")
parser.add_argument("--grasp_tpe_n_startup_trials", type=int, default=None,
                    help="Grasp-level TPE n_startup_trials (default: 3 * n_grasp_sample)")
parser.add_argument("--grasp_tpe_n_ei_candidates", type=int, default=None,
                    help="Grasp-level TPE n_ei_candidates (default: max(24, n_grasp_sample))")
parser.add_argument("--grasp_tpe_multivariate", action="store_true",
                    help="Use multivariate TPE for grasp-level optimization")
parser.add_argument("--grasp_tpe_seed", type=int, default=None,
                    help="Random seed base for grasp-level TPE sampler (default: args_cli.seed)")
parser.add_argument("--grasp_tpe_gamma", type=float, default=None,
                    help="Optional grasp-level TPE gamma (good/bad split fraction, 0 < gamma <= 0.5). Uses internal Optuna API.")

args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest of imports after app launch."""

import csv
import json
import random
from typing import Optional

# Optional matplotlib import for plotting grasp optimization
try:
    import matplotlib.pyplot as plt # type: ignore
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
import numpy as np # type: ignore
import torch # type: ignore

# Set up logging
logger = logging.getLogger(__name__)

from robotouch.urdf_padder import (
    prepare_padded_hands,
    extract_real_joint_values,
    convert_all_padded_hands_to_usd,
)
from nested_multi_sim_config import NestedMultiSimEnv, MultiSimEnvCfg, TASK_NAMES
from grasp_optimizer import BatchGraspOptimizer, GraspParameterRanges


class NestedStatisticsTracker:
    """Track and report statistics for nested grasp optimization."""

    def __init__(self, hand_names: list, tasks: list, n_grasp_iter: int, top_k: int):
        self.hand_names = hand_names
        self.tasks = tasks
        self.n_grasp_iter = n_grasp_iter
        self.top_k = top_k
        self.start_time = time.time()
        self.current_iter = 0

    def get_elapsed_str(self) -> str:
        """Get elapsed time as MM:SS string."""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def print_iteration_progress(self, grasp_iter: int, grasp_optimizer: BatchGraspOptimizer):
        """Print progress for a grasp iteration."""
        self.current_iter = grasp_iter
        elapsed_str = self.get_elapsed_str()
        
        print("=" * 80)
        print(f"Grasp Optimization Progress - Iteration {grasp_iter + 1}/{self.n_grasp_iter}")
        print(f"Time Elapsed: {elapsed_str}")
        print(f"Combos: {len(self.hand_names)} hands x {len(self.tasks)} tasks")
        print()
        
        # Print best scores per combo
        print("Best Grasp Scores per Hand-Object:")
        for hand_idx, hand_name in enumerate(self.hand_names):
            line_parts = [f"  {hand_name}:"]
            for task_idx, task_name in enumerate(self.tasks):
                best_score = grasp_optimizer.optimizer.get_best_score(hand_idx, task_idx)
                task_abbrev = task_name[0].upper()
                if task_name == "stir":
                    task_abbrev = "St"
                elif task_name == "saw":
                    task_abbrev = "Sa"
                score_str = f"{best_score:.3f}" if best_score is not None else "N/A"
                line_parts.append(f"{task_abbrev}:{score_str}")
            print("  ".join(line_parts))
        
        print("=" * 80)

    def print_final_results(self, grasp_optimizer: BatchGraspOptimizer):
        """Print final results."""
        elapsed_str = self.get_elapsed_str()
        
        print("\n" + "=" * 80)
        print("FINAL GRASP OPTIMIZATION RESULTS")
        print(f"Total Time: {elapsed_str}")
        print(f"Grasp Iterations: {self.n_grasp_iter}")
        print("=" * 80)
        
        # Compute and print grasp scores
        print("\nGrasp Scores (avg of top-{} wrench scores):".format(self.top_k))
        grasp_scores = grasp_optimizer.compute_all_grasp_scores(self.top_k)
        for hand_idx, hand_name in enumerate(self.hand_names):
            line_parts = [f"  {hand_name}:"]
            for task_idx, task_name in enumerate(self.tasks):
                score = grasp_scores[(hand_idx, task_idx)]
                task_abbrev = task_name[0].upper()
                if task_name == "stir":
                    task_abbrev = "St"
                elif task_name == "saw":
                    task_abbrev = "Sa"
                line_parts.append(f"{task_abbrev}:{score:.3f}")
            print("  ".join(line_parts))
        
        # Compute and print hand scores
        print("\nHand Scores (avg across all tasks):")
        hand_scores = grasp_optimizer.compute_all_hand_scores(self.top_k)
        sorted_hands = sorted(hand_scores.items(), key=lambda x: x[1], reverse=True)
        for rank, (hand_idx, score) in enumerate(sorted_hands, 1):
            hand_name = self.hand_names[hand_idx]
            print(f"  {rank}. {hand_name}: {score:.4f}")
        
        print("=" * 80)


class NestedGraspStore:
    """Store and save nested grasp optimization results."""

    def __init__(self, output_dir: str, hand_names: list, tasks: list, padded_hands: list, urdf_folder: str):
        self.output_dir = Path(output_dir)
        self.hand_names = hand_names
        self.tasks = tasks
        self.padded_hands = padded_hands
        self.urdf_folder = Path(urdf_folder)

        # Create directory structure
        self.run_dir = self.output_dir / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def save_config(self, args):
        """Save run configuration."""
        config = {
            'urdf_folder': args.urdf_folder,
            'n_grasp_iter': args.n_grasp_iter,
            'n_grasp_sample': args.n_grasp_sample,
            'top_k': args.top_k,
            'device': args.device,
            'tasks': args.tasks,
            'hand_names': self.hand_names,
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.run_dir / "run_config.json", 'w') as f:
            json.dump(config, f, indent=2)

    def save_hand_scores(self, grasp_optimizer: BatchGraspOptimizer, top_k: int):
        """Save hand scores for nested optimizer.
        
        Saves scores.csv to both run_dir and urdf_folder (iteration directory).
        """
        hand_scores = grasp_optimizer.compute_all_hand_scores(top_k)
        
        # Save to run_dir
        scores_path = self.run_dir / "scores.csv"
        self._write_scores_csv(scores_path, hand_scores)
        
        # Always save to urdf_folder (which is the iteration directory in nested optimization)
        if self.urdf_folder.exists():
            iter_scores_path = self.urdf_folder / "scores.csv"
            self._write_scores_csv(iter_scores_path, hand_scores)
            print(f"Scores saved to iteration directory: {iter_scores_path}")
        else:
            print(f"Warning: urdf_folder does not exist: {self.urdf_folder}. Cannot save scores.csv there.")
    
    def _write_scores_csv(self, path: Path, hand_scores: dict):
        """Write hand scores to CSV file."""
        try:
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['hand_index', 'score'])
                for hand_idx in sorted(hand_scores.keys()):
                    score = hand_scores[hand_idx]
                    writer.writerow([hand_idx, f"{score:.6f}"])
            print(f"Hand scores saved to {path}")
        except Exception as e:
            print(f"Error saving scores to {path}: {e}")
            raise
    
    def save_grasp_scores(self, grasp_optimizer: BatchGraspOptimizer, top_k: int):
        """Save detailed grasp scores for all hand-object combinations."""
        grasp_scores = grasp_optimizer.compute_all_grasp_scores(top_k)
        
        csv_path = self.run_dir / "grasp_scores.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['hand_idx', 'hand_name', 'task_idx', 'task_name', 'grasp_score'])
            for (hand_idx, task_idx), score in grasp_scores.items():
                hand_name = self.hand_names[hand_idx]
                task_name = self.tasks[task_idx]
                writer.writerow([hand_idx, hand_name, task_idx, task_name, f"{score:.6f}"])
        print(f"Grasp scores saved to {csv_path}")
    
    def save_optimizer_summary(self, grasp_optimizer: BatchGraspOptimizer):
        """Save optimizer summary to JSON."""
        summary = grasp_optimizer.optimizer.get_summary()
        summary_path = self.run_dir / "optimizer_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"Optimizer summary saved to {summary_path}")
    
    def save_individual_test_results(
        self,
        grasp_iter: int,
        test_scores: torch.Tensor,
        env_to_hand: torch.Tensor,
        env_to_task: torch.Tensor,
        num_combos: int,
        time_per_test: float,
        dt: float,
    ):
        """Save individual test results for a grasp iteration.
        
        Args:
            grasp_iter: Current grasp iteration number (0-indexed)
            test_scores: [num_envs, 12] tensor of individual test scores (raw frame counts)
            env_to_hand: [num_envs] tensor mapping env_idx to hand_idx
            env_to_task: [num_envs] tensor mapping env_idx to task_idx
            num_combos: Number of hand-task combinations
            time_per_test: Time per test in seconds (for normalization)
            dt: Simulation timestep in seconds (for normalization)
        """
        # Create directory for iteration-specific results
        iter_dir = self.run_dir / "individual_test_results" / f"grasp_iter_{grasp_iter:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = iter_dir / "individual_test_results.csv"
        
        # Convert tensors to CPU numpy arrays
        test_scores_np = test_scores.cpu().numpy()  # [num_envs, 12] - raw frame counts
        env_to_hand_np = env_to_hand.cpu().numpy()  # [num_envs]
        env_to_task_np = env_to_task.cpu().numpy()  # [num_envs]
        
        # Normalize test scores: frames held / max frames per test
        max_frames_per_test = time_per_test / dt
        test_scores_normalized = test_scores_np / max_frames_per_test  # [num_envs, 12] - normalized 0-1
        
        num_envs = test_scores_np.shape[0]
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header: grasp_iter, hand_idx, task_idx, sample_idx, f+x, f-x, f+y, f-y, f+z, f-z, t+x, t-x, t+y, t-y, t+z, t-z
            # Scores are normalized (0-1, where 1.0 means held for full test duration)
            # Test names: f+x, f-x, f+y, f-y, f+z, f-z (forces), t+x, t-x, t+y, t-y, t+z, t-z (torques)
            test_names = ['f+x', 'f-x', 'f+y', 'f-y', 'f+z', 'f-z', 't+x', 't-x', 't+y', 't-y', 't+z', 't-z']
            header = ['grasp_iter', 'hand_idx', 'task_idx', 'sample_idx'] + test_names
            writer.writerow(header)
            
            for env_idx in range(num_envs):
                hand_idx = int(env_to_hand_np[env_idx])
                task_idx = int(env_to_task_np[env_idx])
                # sample_idx is which environment within the combo (0, 1, 2, ...)
                sample_idx = env_idx // num_combos
                
                row = [grasp_iter, hand_idx, task_idx, sample_idx] + [
                    f"{test_scores_normalized[env_idx, i]:.6f}" for i in range(12)
                ]
                writer.writerow(row)
        
        print(f"Individual test results saved to {csv_path}")

    def save_grasp_history(self, grasp_iter: int, grasp_optimizer: BatchGraspOptimizer):
        """Save per-iteration grasp configs + scores as a JSON file.

        Saves only the NEW configs from the current iteration (last n_samples_per_combo
        entries from history per combo) to:
            run_dir/individual_test_results/grasp_iter_NNN/grasp_configs.json
        """
        iter_dir = self.run_dir / "individual_test_results" / f"grasp_iter_{grasp_iter:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        n = grasp_optimizer.n_samples_per_combo
        result = {}
        for (hand_idx, obj_idx), entries in grasp_optimizer.optimizer.history.items():
            key = f"hand_{hand_idx}_obj_{obj_idx}"
            # Slice only the entries added in this iteration
            iter_entries = entries[-n:]
            result[key] = [
                {"config": config.to_dict(), "score": score}
                for config, score in iter_entries
            ]

        json_path = iter_dir / "grasp_configs.json"
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Grasp configs saved to {json_path}")

    def save_top_k_grasps(self, grasp_optimizer: BatchGraspOptimizer, top_k: int):
        """Save final top-k grasp configs to run_dir and urdf_folder."""
        top_k_data = grasp_optimizer.optimizer.export_top_k_configs(top_k)

        # Save to run_dir
        run_path = self.run_dir / "top_k_grasps.json"
        with open(run_path, 'w') as f:
            json.dump(top_k_data, f, indent=2)
        print(f"Top-k grasps saved to {run_path}")

        # Also save to urdf_folder (same dual-save pattern as save_hand_scores)
        if self.urdf_folder.exists():
            urdf_path = self.urdf_folder / "top_k_grasps.json"
            with open(urdf_path, 'w') as f:
                json.dump(top_k_data, f, indent=2)
            print(f"Top-k grasps saved to iteration directory: {urdf_path}")
        else:
            print(f"Warning: urdf_folder does not exist: {self.urdf_folder}. Cannot save top_k_grasps.json there.")


def plot_grasp_optimization_progress(grasp_optimizer: BatchGraspOptimizer, output_dir: Path, n_grasp_iter: int, n_grasp_sample: int):
    """Plot grasp optimization score trajectories for each hand-object combination.

    For each (hand, object) combo, we plot:
    - Average score per grasp iteration (one data point per iteration)
    - Best score in that iteration (not cumulative)
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib is required for plotting but is not installed. "
            "Install it (e.g., `pip install matplotlib`) and re-run."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    for hand_idx in range(grasp_optimizer.num_hands):
        for obj_idx in range(grasp_optimizer.num_objects):
            key = (hand_idx, obj_idx)
            history = grasp_optimizer.optimizer.history.get(key, [])
            if not history:
                continue

            # Extract all scores
            scores = [score for _, score in history]
            
            # Group scores by grasp iteration
            # Each iteration has n_grasp_sample trials
            iteration_averages = []
            iteration_best = []
            
            for iter_idx in range(n_grasp_iter):
                start_idx = iter_idx * n_grasp_sample
                end_idx = start_idx + n_grasp_sample
                
                if start_idx < len(scores):
                    iter_scores = scores[start_idx:end_idx]
                    avg_score = sum(iter_scores) / len(iter_scores)
                    best_score = max(iter_scores)  # Best score in this iteration
                    iteration_averages.append(avg_score)
                    iteration_best.append(best_score)
                else:
                    # Not enough trials yet
                    break
            
            if not iteration_averages:
                continue
            
            iterations = list(range(1, len(iteration_averages) + 1))

            fig = plt.figure(figsize=(8, 5))
            plt.plot(iterations, iteration_averages, "o-", label="Avg score per iteration", linewidth=2, markersize=8)
            plt.plot(iterations, iteration_best, "s-", label="Best score in iteration", linewidth=2, markersize=8)
            plt.xlabel("Grasp Iteration")
            plt.ylabel("Grasp Score")
            plt.title(f"Grasp Optimization Progress - Hand {hand_idx}, Obj {obj_idx}")
            plt.grid(True, alpha=0.3)
            plt.legend()

            plot_path = output_dir / f"grasp_progress_hand{hand_idx}_obj{obj_idx}.png"
            try:
                fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            except Exception as e:
                raise RuntimeError(f"Failed to save grasp plot to {plot_path}: {e}") from e
            finally:
                plt.close(fig)

            print(f"Saved grasp optimization plot: {plot_path}")


def set_seed(seed: int):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value. If None, seeds are not set.
    """
    if seed is None:
        return
    
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # PyTorch deterministic operations (slower but more reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to: {seed}")


def run_single_grasp_iteration(env: NestedMultiSimEnv, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run a single grasp iteration (one full episode for all environments).
    
    Returns:
        scores: [num_envs] tensor of wrench test scores (aggregated)
        test_scores: [num_envs, 12] tensor of individual test scores (one per test)
    
    Raises:
        RuntimeError: If the iteration fails to complete or encounters errors
    """
    # Enable single-shot mode for this iteration
    env.cfg.single_shot_mode = True
    env._initial_reset_done = False
    
    # Reset completion status
    # Note: These operations need to work with inference tensors from Isaac Lab
    with torch.inference_mode():
        env.env_completed.zero_()
        env.env_final_scores.zero_()
    
    # Reset all environments
    try:
        # Workaround: Isaac Lab's internal tensors may be in inference mode from previous operations,
        # so we need to call reset() within inference mode to allow modifications
        with torch.inference_mode():
            env.reset()
    except Exception as e:
        error_msg = f"Failed to reset environments in grasp iteration: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e
    
    # Run until all environments complete
    try:
        while not env.all_envs_completed():
            with torch.inference_mode():
                actions = torch.zeros(env.action_space.shape, device=device)
                env.step(actions)
    except Exception as e:
        # Get completion status for detailed error message
        try:
            completion_status = env.get_completion_status()
            error_msg = (
                f"Error during grasp iteration step execution. "
                f"Completion status: {completion_status['completed_count']}/{completion_status['total_count']} "
                f"({completion_status['completed_ratio']*100:.1f}% complete). "
                f"Error: {e}"
            )
        except:
            error_msg = f"Error during grasp iteration step execution: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e
    
    # Verify all environments actually completed
    if not env.all_envs_completed():
        completion_status = env.get_completion_status()
        error_msg = (
            f"Grasp iteration completed but not all environments finished. "
            f"Completion status: {completion_status['completed_count']}/{completion_status['total_count']} "
            f"({completion_status['completed_ratio']*100:.1f}% complete)."
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    # Return final scores and individual test scores
    try:
        with torch.no_grad():
            scores = env.env_final_scores.clone()
            test_scores = env.test_score.clone()  # [num_envs, 12]
            
            # Check for invalid scores (NaN or Inf)
            if torch.isnan(scores).any() or torch.isinf(scores).any():
                nan_count = torch.isnan(scores).sum().item()
                inf_count = torch.isinf(scores).any().item()
                error_msg = (
                    f"Invalid scores detected: {nan_count} NaN values, "
                    f"{'Inf values present' if inf_count else 'no Inf values'}. "
                    f"This indicates a simulation error."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            return scores, test_scores
    except Exception as e:
        error_msg = f"Error retrieving scores from grasp iteration: {e}"
        logger.error(error_msg, exc_info=True)
        raise RuntimeError(error_msg) from e


def main():
    # Set random seed for reproducibility
    set_seed(args_cli.seed)
    
    # Parse tasks
    tasks = [t.strip() for t in args_cli.tasks.split(',')]
    for task in tasks:
        if task not in TASK_NAMES:
            print(f"Error: Unknown task '{task}'. Valid tasks: {TASK_NAMES}")
            return

    print(f"Tasks: {tasks}")
    print(f"URDF folder: {args_cli.urdf_folder}")
    print(f"Grasp iterations: {args_cli.n_grasp_iter}")
    print(f"Grasp samples per combo: {args_cli.n_grasp_sample}")
    print(f"Top-K for averaging: {args_cli.top_k}")

    # Prepare padded hands
    print("Preparing padded hands...")
    padded_hands, joint_template, max_joints = prepare_padded_hands(
        args_cli.urdf_folder,
        output_dir=Path(args_cli.output_dir) / "padded_urdfs"
    )

    if not padded_hands:
        print("Error: No valid hands found in URDF folder")
        return

    hand_names = [h.hand_name for h in padded_hands]
    num_hands = len(padded_hands)
    num_tasks = len(tasks)
    print(f"Found {num_hands} hands: {hand_names}")
    print(f"Max joints: {max_joints}, Joint template size: {len(joint_template)}")

    # Convert padded URDFs to USD (required for MultiUsdFileCfg)
    print("Converting padded URDFs to USD...")
    padded_hands = convert_all_padded_hands_to_usd(
        padded_hands,
        output_dir=str(Path(args_cli.output_dir) / "converted_usd")
    )

    # Update hand_names after conversion (some hands may have failed)
    hand_names = [h.hand_name for h in padded_hands]
    num_hands = len(padded_hands)
    print(f"Successfully converted {num_hands} hands: {hand_names}")

    # Apply GPU compute fraction limit by reducing n_grasp_sample proportionally
    # This directly limits the number of parallel environments (GPU compute load)
    effective_n_grasp_sample = int(args_cli.n_grasp_sample * args_cli.gpu_compute_fraction)
    if effective_n_grasp_sample < 1:
        effective_n_grasp_sample = 1
    if effective_n_grasp_sample != args_cli.n_grasp_sample:
        print(f"GPU compute fraction {args_cli.gpu_compute_fraction*100:.1f}%: reducing n_grasp_sample from {args_cli.n_grasp_sample} to {effective_n_grasp_sample}")
    
    # Calculate total environments
    # Each hand-object combination has n_grasp_sample environments
    num_combos = num_hands * num_tasks
    total_envs = num_combos * effective_n_grasp_sample

    # Create environment config
    env_cfg = MultiSimEnvCfg()
    env_cfg.scene.num_envs = total_envs
    env_cfg.device = args_cli.device
    env_cfg.sim.device = args_cli.device
    env_cfg.single_shot_mode = True  # We run single-shot iterations
    env_cfg.seed = args_cli.seed  # Set seed for deterministic environment creation

    # Prepare config with hand and task info
    env_cfg.prepare_config(
        padded_hands=padded_hands,
        joint_template=joint_template,
        tasks=tasks,
        k=args_cli.top_k,
    )

    print(f"Creating environment with {env_cfg.scene.num_envs} total envs...")
    print(f"  {env_cfg.envs_per_combo} envs per combo x {num_hands} hands x {num_tasks} tasks")

    # Set GPU memory limit before creating environment
    if args_cli.device.startswith('cuda') and args_cli.gpu_memory_fraction < 1.0:
        device_id = int(args_cli.device.split(':')[1]) if ':' in args_cli.device else 0
        torch.cuda.set_per_process_memory_fraction(args_cli.gpu_memory_fraction, device=device_id)
        total_memory_gb = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
        limit_memory_gb = total_memory_gb * args_cli.gpu_memory_fraction
        print(f"Set GPU memory limit to {args_cli.gpu_memory_fraction*100:.1f}% on {args_cli.device}")
        print(f"  Total GPU memory: {total_memory_gb:.2f} GB")
        print(f"  Memory limit: {limit_memory_gb:.2f} GB")
    
    # Set PhysX thread count to limit CPU/GPU compute
    if args_cli.physx_num_threads is not None:
        import os
        # Set environment variable for PhysX thread count (Isaac Sim reads this)
        os.environ['PHYSX_NUM_THREADS'] = str(args_cli.physx_num_threads)
        print(f"Set PhysX CPU thread count to {args_cli.physx_num_threads}")
    
    # Set CUDA compute limiting environment variables
    if args_cli.device.startswith('cuda') and args_cli.gpu_compute_fraction < 1.0:
        import os
        # Limit CUDA stream parallelism (reduces concurrent GPU operations)
        # This is a soft limit but helps reduce GPU compute load
        max_streams = max(1, int(32 * args_cli.gpu_compute_fraction))  # Default is 32 streams
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = str(max_streams)
        print(f"Set CUDA max connections to {max_streams} (compute fraction: {args_cli.gpu_compute_fraction*100:.1f}%)")

    # Create environment (Isaac Sim loads here and stays loaded)
    print("Loading Isaac Sim (this may take a while)...")
    env = NestedMultiSimEnv(cfg=env_cfg, device=args_cli.device)
    print("Isaac Sim loaded successfully!")

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Create grasp optimizer for all hand-object combinations
    print("Initializing grasp optimizer...")
    # Derive grasp-level TPE hyperparameters if not explicitly set
    grasp_n_startup = (
        args_cli.grasp_tpe_n_startup_trials
        if args_cli.grasp_tpe_n_startup_trials is not None
        else max(10, 3 * effective_n_grasp_sample)
    )
    grasp_n_ei = (
        args_cli.grasp_tpe_n_ei_candidates
        if args_cli.grasp_tpe_n_ei_candidates is not None
        else max(24, effective_n_grasp_sample)
    )
    grasp_seed = args_cli.grasp_tpe_seed if args_cli.grasp_tpe_seed is not None else args_cli.seed

    grasp_optimizer = BatchGraspOptimizer(
        num_hands=num_hands,
        num_objects=num_tasks,
        n_samples_per_combo=effective_n_grasp_sample,  # Use effective sample count (after compute fraction)
        param_ranges=GraspParameterRanges(),
        seed=grasp_seed,
        n_startup_trials=grasp_n_startup,
        n_ei_candidates=grasp_n_ei,
        multivariate=args_cli.grasp_tpe_multivariate,
        gamma=args_cli.grasp_tpe_gamma,
    )

    # Create statistics tracker and store
    stats = NestedStatisticsTracker(hand_names, tasks, args_cli.n_grasp_iter, args_cli.top_k)
    store = NestedGraspStore(args_cli.output_dir, hand_names, tasks, padded_hands, args_cli.urdf_folder)
    store.save_config(args_cli)
    
    # Set up logging to file and console
    log_file = store.run_dir / "grasp_optimization.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing logging configuration
    )
    logger.info(f"Logging initialized. Log file: {log_file}")

    # Signal handling for graceful shutdown
    should_stop = False

    def signal_handler(signum, frame):
        nonlocal should_stop
        print("\nReceived interrupt signal. Saving results and exiting...")
        should_stop = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n" + "=" * 80)
    print("STARTING NESTED GRASP OPTIMIZATION")
    print(f"Isaac Sim is loaded and will stay running for all {args_cli.n_grasp_iter} iterations")
    print("=" * 80 + "\n")

    # Main grasp optimization loop
    for grasp_iter in range(args_cli.n_grasp_iter):
        if should_stop:
            print("Stopping early due to interrupt...")
            break
        
        print(f"\n--- Grasp Iteration {grasp_iter + 1}/{args_cli.n_grasp_iter} ---")
        logger.info(f"Starting grasp iteration {grasp_iter + 1}/{args_cli.n_grasp_iter}")
        
        try:
            # Step 1: Sample grasp configs from all TPE optimizers
            print("Sampling grasp configurations from TPE optimizers...")
            configs = grasp_optimizer.sample_all_configs()
            
            # Step 2: Convert configs to tensors for environment
            config_tensors = grasp_optimizer.configs_to_tensors(configs, device=args_cli.device)
            
            # Step 3: Set grasp configs in environment
            env.set_grasp_configs(
                obj_dist_palm=config_tensors['obj_dist_palm'],
                obj_angle_palm=config_tensors['obj_angle_palm'],
                finger_spread=config_tensors['finger_spread'],
                finger_m_rand=config_tensors['finger_m_rand'],
            )
            
            # Step 4: Run single grasp iteration (all environments in parallel)
            print("Running wrench tests...")
            scores, test_scores = run_single_grasp_iteration(env, args_cli.device)
            
            # Step 5: Save individual test results for this iteration
            print("Saving individual test results...")
            store.save_individual_test_results(
                grasp_iter=grasp_iter,
                test_scores=test_scores,
                env_to_hand=env.env_to_hand,
                env_to_task=env.env_to_task,
                num_combos=num_combos,
                time_per_test=env.cfg.time_per_test,
                dt=env.cfg.dt,
            )
            
            # Step 6: Update TPE optimizers with results
            print("Updating TPE optimizers with results...")
            grasp_optimizer.update_all_with_results(configs, scores)

            # Step 7: Save grasp configs for this iteration
            store.save_grasp_history(grasp_iter, grasp_optimizer)

            # Print progress
            stats.print_iteration_progress(grasp_iter, grasp_optimizer)

            # Plot grasp optimization progress after each grasp iteration
            plots_dir = store.run_dir / "grasp_plots"
            plot_grasp_optimization_progress(grasp_optimizer, plots_dir, args_cli.n_grasp_iter, args_cli.n_grasp_sample)
            
            logger.info(f"Successfully completed grasp iteration {grasp_iter + 1}/{args_cli.n_grasp_iter}")
            
        except RuntimeError as e:
            # Critical error - stop the entire process
            error_msg = (
                f"\n{'='*80}\n"
                f"CRITICAL ERROR in grasp iteration {grasp_iter + 1}/{args_cli.n_grasp_iter}\n"
                f"{'='*80}\n"
                f"Error: {e}\n"
                f"Timestamp: {datetime.now().isoformat()}\n"
                f"\nThe optimization process will now STOP.\n"
                f"Please check the logs and fix the issue before resuming.\n"
                f"{'='*80}\n"
            )
            logger.critical(error_msg, exc_info=True)
            print(error_msg)
            
            # Save error state before exiting
            try:
                error_log_path = store.run_dir / f"error_log_iter_{grasp_iter:03d}.txt"
                with open(error_log_path, 'w') as f:
                    f.write(error_msg)
                    f.write(f"\nFull traceback:\n")
                    f.write(traceback.format_exc())
                logger.info(f"Error log saved to: {error_log_path}")
            except Exception as save_error:
                logger.error(f"Failed to save error log: {save_error}")
            
            # Stop the process
            sys.exit(1)
            
        except Exception as e:
            # Unexpected error - also stop
            error_msg = (
                f"\n{'='*80}\n"
                f"UNEXPECTED ERROR in grasp iteration {grasp_iter + 1}/{args_cli.n_grasp_iter}\n"
                f"{'='*80}\n"
                f"Error type: {type(e).__name__}\n"
                f"Error: {e}\n"
                f"Timestamp: {datetime.now().isoformat()}\n"
                f"\nThe optimization process will now STOP.\n"
                f"Please check the logs and fix the issue before resuming.\n"
                f"{'='*80}\n"
            )
            logger.critical(error_msg, exc_info=True)
            print(error_msg)
            
            # Save error state before exiting
            try:
                error_log_path = store.run_dir / f"error_log_iter_{grasp_iter:03d}.txt"
                with open(error_log_path, 'w') as f:
                    f.write(error_msg)
                    f.write(f"\nFull traceback:\n")
                    f.write(traceback.format_exc())
                logger.info(f"Error log saved to: {error_log_path}")
            except Exception as save_error:
                logger.error(f"Failed to save error log: {save_error}")
            
            # Stop the process
            sys.exit(1)

    # Compute final scores
    print("\nComputing final grasp and hand scores...")
    
    # Print final results
    stats.print_final_results(grasp_optimizer)

    # Save results
    print("\nSaving final results...")
    store.save_hand_scores(grasp_optimizer, args_cli.top_k)
    store.save_grasp_scores(grasp_optimizer, args_cli.top_k)
    store.save_optimizer_summary(grasp_optimizer)
    store.save_top_k_grasps(grasp_optimizer, args_cli.top_k)

    # Plot grasp optimization score trajectories
    plots_dir = store.run_dir / "grasp_plots"
    plot_grasp_optimization_progress(grasp_optimizer, plots_dir, args_cli.n_grasp_iter, args_cli.n_grasp_sample)

    print(f"\nResults saved to: {store.run_dir}")
    total_elapsed = time.time() - script_start_time
    print(f"Total script runtime: {total_elapsed:.1f}s")

    def _shutdown(timeout_s: float = 0.1) -> None:
        """Attempt graceful shutdown; force exit if cleanup hangs."""
        def _close_all():
            try:
                env.close()
            except Exception as exc:
                print(f"Warning: env.close() failed: {exc}")
            try:
                simulation_app.close()
            except Exception as exc:
                print(f"Warning: simulation_app.close() failed: {exc}")

        thread = threading.Thread(target=_close_all, daemon=True)
        thread.start()
        thread.join(timeout=timeout_s)
        if thread.is_alive():
            total_elapsed = time.time() - script_start_time
            print(f"Total script runtime: {total_elapsed:.1f}s")
            print("Shutdown is taking too long; forcing exit.")
            os._exit(0)
        sys.exit(0)

    _shutdown()


if __name__ == "__main__":
    main()
