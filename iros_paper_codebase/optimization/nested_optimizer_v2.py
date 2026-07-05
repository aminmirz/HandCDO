"""
Nested TPE Optimization Script for Hand Generation

This script implements a two-level nested optimization:
- Outer loop: Hand-level TPE optimization (n_hand_iter iterations)
- Inner loop: Grasp-level TPE optimization (run inside nested_multi_grasp_selector.py)

The nested simulation script handles grasp optimization internally while keeping
Isaac Sim loaded, outputting a scores.csv file with per-hand scores that this
script uses to update the hand-level TPE optimizer.
"""
from datetime import datetime

import optuna # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import itertools
import subprocess
import os
import sys
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union, List
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

# Lock for thread-safe access to parameters.npz
_params_file_lock = threading.Lock()

# Add the project root to Python path to allow imports from generation
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add the generation directory itself so legacy modules like HandClass.py,
# which do "from Config import *", can still find Config.py as a top-level module.
generation_dir = project_root / "generation"
if str(generation_dir) not in sys.path:
    sys.path.insert(0, str(generation_dir))

from generation.Config import PalmConfig, FingerConfig, HandConfig
from generation.HandClass import Hand

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
OPTIMIZATION_DIR = Path(__file__).parent.resolve()
DEFAULT_SIMULATION_SCRIPT = str(OPTIMIZATION_DIR / "grasp_evaluation" / "nested_multi_grasp_selector.py")

# Matplotlib is REQUIRED for plotting. We still import it in a try/except so that
# we can raise a clear, actionable error at runtime if it's missing.
try:
    import matplotlib.pyplot as plt # type: ignore
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.error("matplotlib not available; plotting is required and will fail.")


@dataclass
class NestedOptimizationConfig:
    """Configuration for the nested optimization process."""
    base_hand_dir: str = "generation/base_hand/hand_0"
    optimization_root: str = "./nested_hand_optimization"
    simulation_script: str = DEFAULT_SIMULATION_SCRIPT
    n_hand_iter: int = 300  # Number of hand-level optimization iterations
    n_samples_per_iter: int = 8  # Number of hands per iteration (exploitation phase)
    n_explore_samples_per_iter: int = 8  # Number of hands per iteration (structural exploration phase)
    study_name: str = "nested_hand_optimization"
    direction: str = "maximize"  # "minimize" or "maximize"
    n_jobs: int = 1  # Number of parallel jobs
    n_parallel_hands: int = 8  # Number of hands to generate in parallel per iteration
    score_column: str = "score"  # Column name in scores.csv
    resume: bool = False  # Whether to resume from a previous run

    # Hand-level TPE optimizer hyperparameters
    hand_tpe_n_startup_trials: Optional[int] = None  # If None, use 3 * n_samples_per_iter
    hand_tpe_n_ei_candidates: Optional[int] = None   # If None, use max(24, n_samples_per_iter)
    hand_tpe_multivariate: bool = False              # Whether to use multivariate TPE at hand level
    hand_tpe_seed: Optional[int] = 0                 # Seed for hand-level TPE (None = random)
    hand_tpe_gamma: Optional[float] = None           # Optional good/bad split fraction (0–0.5). Uses internal Optuna API.
    
    # Grasp optimization parameters (passed to nested_multi_grasp_selector.py)
    n_grasp_iter: int = 64  # Number of grasp optimization iterations
    n_grasp_sample: int = 16  # Number of grasp samples per hand-object combo
    top_k: int = 8  # Top-K grasps to average for final score
    tasks: str = "hammer,stir,knife"  # Comma-separated list of object types (same as original)
    device: str = "cuda:0"  # Device for simulation
    gpu_memory_fraction: float = 1.0  # Fraction of GPU memory to use (0.0 to 1.0, e.g., 0.5 for 50%)
    gpu_compute_fraction: float = 1.0  # Fraction of GPU compute to use (0.0 to 1.0, e.g., 0.5 for 50%). Limits parallel environments.
    physx_num_threads: Optional[int] = None  # Number of PhysX CPU threads (None = auto). Lower values reduce CPU/GPU load.
    
    # Grasp-level TPE optimizer hyperparameters (passed to nested_multi_grasp_selector.py)
    grasp_tpe_n_startup_trials: Optional[int] = None  # If None, use 3 * n_grasp_sample
    grasp_tpe_n_ei_candidates: Optional[int] = None   # If None, use max(24, n_grasp_sample)
    grasp_tpe_multivariate: bool = False              # Whether to use multivariate TPE at grasp level
    grasp_tpe_seed: Optional[int] = 0                 # Seed base for grasp-level TPE (None = random)
    grasp_tpe_gamma: Optional[float] = None           # Optional good/bad split fraction (0–0.5). Uses internal Optuna API.
    
    # Parameter ranges (from Config.py defaults)
    # Can be a single tuple (applied to all) or a list of tuples (one per finger/thumb)
    # finger_angle_ranges: Union[Tuple[float, float], list[Tuple[float, float]]] = field(default_factory=lambda: [(0.0, 10.0), (0.0, 0.0), (-10.0, 0.0)])
    # thumb_angle_ranges: Union[Tuple[float, float], list[Tuple[float, float]]] = field(default_factory=lambda: (-30.0, 30.0))
    # finger_normal_offset_ranges: Union[Tuple[float, float], list[Tuple[float, float]]] = field(default_factory=lambda: [(0.0, 5.0), (0.0, 10.0), (0.0, 5.0)])
    # thumb_normal_offset_ranges: Union[Tuple[float, float], list[Tuple[float, float]]] = field(default_factory=lambda: (-20.0, 20.0))
    # finger_side_offset_ranges: Union[Tuple[float, float], list[Tuple[float, float]]] = field(default_factory=lambda: [(0.0, 20.0), (0.0, 0.0), (-20.0, 0.0)])
    # thumb_side_offset_ranges: Union[Tuple[float, float], list[Tuple[float, float]]] = field(default_factory=lambda: (-20.0, 0.0))
    # palm_bump_max_height_intensity_mm_range: Tuple[float, float] = (0.0, 15.0)
    # palm_bumps_spread_range: Tuple[float, float] = (0.1, 0.5)
    # palm_bump_center_angle_deg_range: Tuple[float, float] = (0.0, 360.0)
    # palm_bump_center_offset_range: Tuple[float, float] = (0.0, 1.0)
    
    finger_angle_ranges: Union[Tuple[float, float], list[Tuple[float, float]]] = field(default_factory=lambda: [(0.0, 30.0), (0.0, 0.0), (-30.0, 0.0)])
    thumb_angle_ranges: Union[Tuple[float, float], list[Tuple[float, float]]] = field(default_factory=lambda: (-30.0, 30.0))
    finger_normal_offset_ranges: Union[Tuple[float, float], list[Tuple[float, float]]] = field(default_factory=lambda: [(0.0, 5.0), (0.0, 10.0), (0.0, 5.0)])
    thumb_normal_offset_ranges: Union[Tuple[float, float], list[Tuple[float, float]]] = field(default_factory=lambda: (-30.0, 30.0))
    finger_side_offset_ranges: Union[Tuple[float, float], list[Tuple[float, float]]] = field(default_factory=lambda: [(0.0, 30.0), (0.0, 0.0), (-30.0, 0.0)])
    thumb_side_offset_ranges: Union[Tuple[float, float], list[Tuple[float, float]]] = field(default_factory=lambda: (-40.0, 10.0))
    palm_bump_max_height_intensity_mm_range: Tuple[float, float] = (0.0, 20.0)
    palm_bumps_spread_range: Tuple[float, float] = (0.05, 0.3)
    palm_bump_center_angle_deg_range: Tuple[float, float] = (0.0, 360.0)
    palm_bump_center_offset_range: Tuple[float, float] = (0.0, 1.0)
    palm_bump_intensity_ratio_range: Tuple[float, float] = (0.0, 1.0)

    # Fingertip scale ranges (shared across all fingers & thumbs)
    fingertip_scale_y_range: Tuple[float, float] = (1.0, 1.5)
    fingertip_scale_z_range: Tuple[float, float] = (0.5, 1.5)

    # Link added length ranges (one value per finger/thumb, replicated to all links)
    finger_link_added_length_range: Tuple[int, int] = (0, 10)
    thumb_link_added_length_range: Tuple[int, int] = (0, 10)

    # Finger/thumb structure code options (categorical)
    finger_code_options: list = field(default_factory=lambda: ["110-11-0101010101", "000-03-0110010101"])
    thumb_code_options: list = field(default_factory=lambda: ["1-2-10101010", "0-2-10101010"])

    # Variable finger count options (categorical)
    finger_number_options: list = field(default_factory=lambda: [2, 3])
    finger_index2drop_options: list = field(default_factory=lambda: [2])

    # Number of fingers and thumbs (max from base hand; controls loop counts for sampling)
    max_finger_number: int = 3
    thumb_number: int = 1


def define_search_space(trial: optuna.Trial, config: NestedOptimizationConfig) -> Dict[str, Any]:
    """
    Define the Optuna search space for hand parameters.
    
    Args:
        trial: Optuna trial object
        config: Optimization configuration
        
    Returns:
        Dictionary of sampled parameters
    """
    params = {}
    
    # Helper function to get range for index i
    def get_range(ranges, index, default_range):
        if isinstance(ranges, tuple):
            return ranges
        elif isinstance(ranges, list):
            if index < len(ranges):
                return ranges[index]
            else:
                return default_range
        else:
            return default_range
    
    # Finger angles (always sample for max_finger_number fingers)
    for i in range(config.max_finger_number):
        angle_range = get_range(config.finger_angle_ranges, i, (-5.0, 5.0))
        params[f'angle_finger_{i}'] = trial.suggest_float(
            f'angle_finger_{i}',
            angle_range[0],
            angle_range[1]
        )

    # Thumb angles
    for i in range(config.thumb_number):
        angle_range = get_range(config.thumb_angle_ranges, i, (-5.0, 5.0))
        params[f'angle_thumb_{i}'] = trial.suggest_float(
            f'angle_thumb_{i}',
            angle_range[0],
            angle_range[1]
        )

    # Normal offsets for fingers
    for i in range(config.max_finger_number):
        offset_range = get_range(config.finger_normal_offset_ranges, i, (-5.0, 5.0))
        params[f'normal_offset_finger_{i}'] = trial.suggest_float(
            f'normal_offset_finger_{i}',
            offset_range[0],
            offset_range[1]
        )

    # Normal offsets for thumbs
    for i in range(config.thumb_number):
        offset_range = get_range(config.thumb_normal_offset_ranges, i, (-5.0, 0.0))
        params[f'normal_offset_thumb_{i}'] = trial.suggest_float(
            f'normal_offset_thumb_{i}',
            offset_range[0],
            offset_range[1]
        )

    # Side offsets for fingers
    for i in range(config.max_finger_number):
        offset_range = get_range(config.finger_side_offset_ranges, i, (-1.0, 1.0))
        params[f'side_offset_finger_{i}'] = trial.suggest_float(
            f'side_offset_finger_{i}',
            offset_range[0],
            offset_range[1]
        )

    # Side offsets for thumbs
    for i in range(config.thumb_number):
        offset_range = get_range(config.thumb_side_offset_ranges, i, (-1.0, 1.0))
        params[f'side_offset_thumb_{i}'] = trial.suggest_float(
            f'side_offset_thumb_{i}',
            offset_range[0],
            offset_range[1]
        )

    # Palm bump parameters (2 bumps)
    params['palm_bump_max_height_intensity_mm'] = trial.suggest_float(
        'palm_bump_max_height_intensity_mm',
        config.palm_bump_max_height_intensity_mm_range[0],
        config.palm_bump_max_height_intensity_mm_range[1]
    )
    for b in range(2):
        params[f'palm_bump_{b}_spread'] = trial.suggest_float(
            f'palm_bump_{b}_spread',
            config.palm_bumps_spread_range[0],
            config.palm_bumps_spread_range[1]
        )
        params[f'palm_bump_{b}_center_angle_deg'] = trial.suggest_float(
            f'palm_bump_{b}_center_angle_deg',
            config.palm_bump_center_angle_deg_range[0],
            config.palm_bump_center_angle_deg_range[1]
        )
        params[f'palm_bump_{b}_center_offset'] = trial.suggest_float(
            f'palm_bump_{b}_center_offset',
            config.palm_bump_center_offset_range[0],
            config.palm_bump_center_offset_range[1]
        )
        params[f'palm_bump_{b}_intensity_ratio'] = trial.suggest_float(
            f'palm_bump_{b}_intensity_ratio',
            config.palm_bump_intensity_ratio_range[0],
            config.palm_bump_intensity_ratio_range[1]
        )

    # Fingertip scale parameters (shared across all fingers & thumbs)
    params['fingertip_scale_y'] = trial.suggest_float(
        'fingertip_scale_y',
        config.fingertip_scale_y_range[0],
        config.fingertip_scale_y_range[1]
    )
    params['fingertip_scale_z'] = trial.suggest_float(
        'fingertip_scale_z',
        config.fingertip_scale_z_range[0],
        config.fingertip_scale_z_range[1]
    )

    # Link added length per finger (one value replicated to all links in that finger)
    for i in range(config.max_finger_number):
        params[f'link_added_length_finger_{i}'] = trial.suggest_int(
            f'link_added_length_finger_{i}',
            config.finger_link_added_length_range[0],
            config.finger_link_added_length_range[1]
        )

    # Link added length per thumb
    for i in range(config.thumb_number):
        params[f'link_added_length_thumb_{i}'] = trial.suggest_int(
            f'link_added_length_thumb_{i}',
            config.thumb_link_added_length_range[0],
            config.thumb_link_added_length_range[1]
        )

    # NOTE: Structural parameters (finger_code, thumb_code, finger_number,
    # finger_index2drop) are NOT sampled here. They are sampled once per
    # iteration in run_nested_optimization() and injected into all hands'
    # params dicts. This ensures all hands in one simulation batch share
    # identical URDF topology, which is required by PhysX ArticulationView.

    return params


def params_to_hand_config(params: Dict[str, Any], base_hand_dir: str) -> Tuple[PalmConfig, list, list, HandConfig]:
    """
    Convert Optuna trial parameters to hand configuration.
    
    Args:
        params: Dictionary of parameters from Optuna trial
        base_hand_dir: Path to base hand configuration directory
        
    Returns:
        Tuple of (palm_cfg, finger_cfgs, thumb_cfgs, hand_cfg)
    """
    # Convert to absolute path if relative
    base_hand_path = Path(base_hand_dir)
    if not base_hand_path.is_absolute():
        # Resolve relative to project root
        project_root = Path(__file__).parent.parent
        base_hand_path = (project_root / base_hand_dir).resolve()
    
    base_hand_dir = str(base_hand_path)
    
    # Verify the directory exists
    if not os.path.exists(base_hand_dir):
        raise ValueError(f"Base hand directory does not exist: {base_hand_dir}")
    
    # 1. Load ALL base configs (always load max fingers from base hand)
    palm_cfg_path = os.path.join(base_hand_dir, "palm_cfg.py")
    if not os.path.exists(palm_cfg_path):
        raise ValueError(f"Palm config file does not exist: {palm_cfg_path}")

    palm_cfg = PalmConfig(data=palm_cfg_path)
    base_finger_number = palm_cfg.finger_number  # Original count from base hand
    finger_cfgs = [
        FingerConfig(data=os.path.join(base_hand_dir, f"finger_cfg_{i}.py"))
        for i in range(base_finger_number)
    ]
    thumb_cfgs = [
        FingerConfig(data=os.path.join(base_hand_dir, f"thumb_cfg_{i}.py"))
        for i in range(palm_cfg.thumb_number)
    ]
    hand_cfg = HandConfig(data=os.path.join(base_hand_dir, "hand_cfg.py"))

    # 2. Apply finger/thumb structure codes
    finger_code = params.get('finger_code')
    thumb_code = params.get('thumb_code')
    if finger_code is not None:
        for cfg in finger_cfgs:
            cfg.code = finger_code
            cfg.code_data()
            cfg.generate()
    if thumb_code is not None:
        for cfg in thumb_cfgs:
            cfg.code = thumb_code
            cfg.code_data()
            cfg.generate()

    # 3. Apply fingertip scale factor (shared across all fingers & thumbs)
    scale_y = params.get('fingertip_scale_y', 1.0)
    scale_z = params.get('fingertip_scale_z', 1.0)
    fingertip_scale = (1.0, scale_y, scale_z)
    for cfg in finger_cfgs:
        cfg.fingertip_scale_factor = fingertip_scale
    for cfg in thumb_cfgs:
        cfg.fingertip_scale_factor = fingertip_scale

    # 4. Apply link added length (one value per finger/thumb, replicated to all links)
    for i, cfg in enumerate(finger_cfgs):
        link_len = float(params.get(f'link_added_length_finger_{i}', 0))
        cfg.link_added_length_mm_list = [link_len] * 6  # 6 links per finger
    for i, cfg in enumerate(thumb_cfgs):
        link_len = float(params.get(f'link_added_length_thumb_{i}', 0))
        cfg.link_added_length_mm_list = [link_len] * 5  # 5 links per thumb

    # 5. Apply finger/thumb angles and offsets (loop over all loaded configs)
    for i in range(len(finger_cfgs)):
        if i < len(palm_cfg.finger_angle_deg_list):
            palm_cfg.finger_angle_deg_list[i] = params[f'angle_finger_{i}']
        else:
            palm_cfg.finger_angle_deg_list.append(params[f'angle_finger_{i}'])

    for i in range(palm_cfg.thumb_number):
        if i < len(palm_cfg.thumb_angle_deg_list):
            palm_cfg.thumb_angle_deg_list[i] = params[f'angle_thumb_{i}']
        else:
            palm_cfg.thumb_angle_deg_list.append(params[f'angle_thumb_{i}'])

    for i in range(len(finger_cfgs)):
        if i < len(palm_cfg.finger_base_normal_offset_mm_list):
            palm_cfg.finger_base_normal_offset_mm_list[i] = params[f'normal_offset_finger_{i}']
        else:
            palm_cfg.finger_base_normal_offset_mm_list.append(params[f'normal_offset_finger_{i}'])

    for i in range(palm_cfg.thumb_number):
        if i < len(palm_cfg.thumb_base_normal_offset_mm_list):
            palm_cfg.thumb_base_normal_offset_mm_list[i] = params[f'normal_offset_thumb_{i}']
        else:
            palm_cfg.thumb_base_normal_offset_mm_list.append(params[f'normal_offset_thumb_{i}'])

    for i in range(len(finger_cfgs)):
        if i < len(palm_cfg.finger_base_side_offset_mm_list):
            palm_cfg.finger_base_side_offset_mm_list[i] = params[f'side_offset_finger_{i}']
        else:
            palm_cfg.finger_base_side_offset_mm_list.append(params[f'side_offset_finger_{i}'])

    for i in range(palm_cfg.thumb_number):
        if i < len(palm_cfg.thumb_base_side_offset_mm_list):
            palm_cfg.thumb_base_side_offset_mm_list[i] = params[f'side_offset_thumb_{i}']
        else:
            palm_cfg.thumb_base_side_offset_mm_list.append(params[f'side_offset_thumb_{i}'])

    # 6. Set 2-bump palm parameters
    palm_cfg.bumps_number = 2
    palm_cfg.bump_max_height_intensity_mm = params['palm_bump_max_height_intensity_mm']
    palm_cfg.bump_height_intensity_list = [
        params.get('palm_bump_0_intensity_ratio', 1.0),
        params.get('palm_bump_1_intensity_ratio', 1.0),
    ]
    palm_cfg.bumps_spread_list = [
        params.get('palm_bump_0_spread', 0.25),
        params.get('palm_bump_1_spread', 0.25),
    ]
    palm_cfg.bumps_aspect_ratio_list = [1.0, 1.0]
    palm_cfg.bump_rotation_deg_list = [0.0, 0.0]
    palm_cfg.bump_center_angle_deg_list = [
        params.get('palm_bump_0_center_angle_deg', 0.0),
        params.get('palm_bump_1_center_angle_deg', 0.0),
    ]
    palm_cfg.bump_center_offset_list = [
        params.get('palm_bump_0_center_offset', 0.0),
        params.get('palm_bump_1_center_offset', 0.0),
    ]

    # 7. Update palm configuration (recalculate finger/thumb positions for all 3 fingers)
    palm_cfg.update()

    # 8. AFTER update(): set finger_number and finger_index2drop
    # Hand constructor handles dropping excess configs
    opt_finger_number = params.get('finger_number')
    if opt_finger_number is not None:
        palm_cfg.finger_number = int(opt_finger_number)
    opt_finger_index2drop = params.get('finger_index2drop')
    if opt_finger_index2drop is not None:
        palm_cfg.finger_index2drop = int(opt_finger_index2drop)

    return palm_cfg, finger_cfgs, thumb_cfgs, hand_cfg


def _generate_single_hand(trial: optuna.Trial, params: Dict[str, Any], iteration: int, hand_index: int, config: NestedOptimizationConfig, trial_counter: dict) -> Tuple[optuna.Trial, Dict[str, Any], int, Optional[Exception]]:
    """
    Helper function to generate a single hand model (for parallel execution).
    
    Returns:
        Tuple of (trial, params, hand_index, error) where error is None if successful
    """
    try:
        generate_hand_models(params, iteration, hand_index, config)
        save_parameters(params, iteration, hand_index, config)
        trial_counter['count'] += 1
        total_trials = config.n_hand_iter * config.n_samples_per_iter
        logger.info(f"Generated hand {hand_index} for iteration {iteration} (trial {trial_counter['count']}/{total_trials})")
        return (trial, params, hand_index, None)
    except Exception as e:
        logger.error(f"Error generating hand {hand_index}: {e}")
        trial_counter['count'] += 1
        return (trial, params, hand_index, e)


def generate_hand_models(params: Dict[str, Any], iteration: int, hand_index: int, config: NestedOptimizationConfig) -> str:
    """
    Generate hand models for a trial.
    
    Args:
        params: Dictionary of parameters
        iteration: Current iteration number
        hand_index: Index of hand in current iteration
        config: Optimization configuration
        
    Returns:
        Path to generated hand directory
    """
    # Create output directory
    iter_dir = os.path.join(config.optimization_root, f"iter_{iteration}")
    hand_dir = os.path.join(iter_dir, f"hand_{hand_index}")
    
    if not os.path.exists(hand_dir):
        os.makedirs(hand_dir)
    
    # Convert parameters to hand configuration
    palm_cfg, finger_cfgs, thumb_cfgs, hand_cfg = params_to_hand_config(params, config.base_hand_dir)
    
    # Save configurations
    palm_cfg.save_config(os.path.join(hand_dir, "palm_cfg.py"))
    for i, finger_cfg in enumerate(finger_cfgs):
        finger_cfg.save_config(os.path.join(hand_dir, f"finger_cfg_{i}.py"))
    for i, thumb_cfg in enumerate(thumb_cfgs):
        thumb_cfg.save_config(os.path.join(hand_dir, f"thumb_cfg_{i}.py"))
    hand_cfg.save_config(os.path.join(hand_dir, "hand_cfg.py"))
    
    # Generate hand using Hand class
    hand = Hand(
        palm_cfg=palm_cfg,
        finger_cfgs=finger_cfgs,
        thumb_cfgs=thumb_cfgs,
        hand_cfg=hand_cfg,
        root_dir=hand_dir
    )
    hand.blender_full_assembly()
    
    logger.info(f"Generated hand model at {hand_dir}")
    return hand_dir


def save_parameters(params: Dict[str, Any], iteration: int, hand_index: int, config: NestedOptimizationConfig):
    """
    Save parameters to parameters.npz file.
    
    Args:
        params: Dictionary of parameters
        iteration: Current iteration number
        hand_index: Index of hand in current iteration
        config: Optimization configuration
    """
    params_file = os.path.join(config.optimization_root, "parameters.npz")

    # Key format: iter_{iteration:02d}_{hand_index:02d}
    key = f"iter_{iteration:02d}_{hand_index:02d}"

    # Thread-safe read-modify-write (parallel hand generation writes concurrently)
    with _params_file_lock:
        # Load existing parameters if file exists
        if os.path.exists(params_file):
            loaded = np.load(params_file, allow_pickle=True)
            # Convert to regular dict, handling both array and dict values
            existing_params = {}
            for k in loaded.keys():
                val = loaded[k]
                # If it's a numpy array with a single element that's a dict, extract it
                if isinstance(val, np.ndarray) and val.dtype == object:
                    existing_params[k] = val.item() if val.size == 1 else val
                else:
                    existing_params[k] = val
        else:
            existing_params = {}

        # Add new parameters
        existing_params[key] = params

        # Save all parameters using savez_compressed with pickle support
        # Convert dict values to numpy arrays for saving
        save_dict = {}
        for k, v in existing_params.items():
            if isinstance(v, dict):
                # Store dict as object array (requires pickle)
                save_dict[k] = np.array([v], dtype=object)
            else:
                save_dict[k] = v

        np.savez_compressed(params_file, **save_dict)
    logger.debug(f"Saved parameters with key: {key}")


def check_system_resources():
    """Check system resources and log diagnostics."""
    try:
        import psutil # type: ignore
        # Memory info
        mem = psutil.virtual_memory()
        logger.info(f"System memory: {mem.used / (1024**3):.2f} GB used / {mem.total / (1024**3):.2f} GB total ({mem.percent:.1f}%)")
        if mem.percent > 90:
            logger.warning(f"System memory usage is high ({mem.percent:.1f}%) - may cause OOM kills")
        
        # GPU info (if available)
        try:
            import torch # type: ignore
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    mem_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    mem_total = props.total_memory / (1024**3)
                    logger.info(f"GPU {i} ({props.name}): {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved, {mem_total:.2f} GB total")
        except Exception:
            pass
    except ImportError:
        logger.warning("psutil not available - cannot check system resources")
    except Exception as e:
        logger.warning(f"Error checking system resources: {e}")


def run_nested_simulation(iteration: int, config: NestedOptimizationConfig) -> bool:
    """
    Execute nested simulation script via subprocess.
    
    The nested simulation script runs grasp-level TPE optimization internally
    with Isaac Sim staying loaded, then outputs scores.csv with per-hand scores.
    
    Args:
        iteration: Current iteration number
        config: Optimization configuration
        
    Returns:
        True if simulation succeeded, False otherwise
    """
    if not config.simulation_script:
        logger.warning("No simulation script specified, skipping simulation")
        return True
    
    iter_dir = os.path.join(config.optimization_root, f"iter_{iteration}")
    output_dir = os.path.join(iter_dir, "simulation_output")
    
    try:
        simulation_script = Path(config.simulation_script)
        if not simulation_script.is_absolute():
            simulation_script = (OPTIMIZATION_DIR / simulation_script).resolve()
        simulation_script_str = str(simulation_script)

        # Build command for nested_multi_grasp_selector.py
        cmd = [simulation_script_str]
        if simulation_script_str.endswith('.py'):
            cmd = ["python", simulation_script_str]
        
        # Core arguments
        cmd.extend([
            "--urdf_folder", iter_dir,
            "--output_dir", output_dir,
            "--device", config.device,
            "--tasks", config.tasks,
        ])
        
        # Grasp optimization parameters
        cmd.extend([
            "--n_grasp_iter", str(config.n_grasp_iter),
            "--n_grasp_sample", str(config.n_grasp_sample),
            "--top_k", str(config.top_k),
        ])
        # Grasp-level TPE hyperparameters (only pass if explicitly set to avoid breaking older scripts)
        if config.grasp_tpe_n_startup_trials is not None:
            cmd.extend([
                "--grasp_tpe_n_startup_trials", str(config.grasp_tpe_n_startup_trials),
            ])
        if config.grasp_tpe_n_ei_candidates is not None:
            cmd.extend([
                "--grasp_tpe_n_ei_candidates", str(config.grasp_tpe_n_ei_candidates),
            ])
        if config.grasp_tpe_multivariate:
            cmd.append("--grasp_tpe_multivariate")
        if config.grasp_tpe_seed is not None:
            cmd.extend([
                "--grasp_tpe_seed", str(config.grasp_tpe_seed),
            ])
        if config.grasp_tpe_gamma is not None:
            cmd.extend([
                "--grasp_tpe_gamma", str(config.grasp_tpe_gamma),
            ])
        
        # GPU memory and compute limits
        if config.gpu_memory_fraction < 1.0:
            cmd.extend([
                "--gpu_memory_fraction", str(config.gpu_memory_fraction),
            ])
        if config.gpu_compute_fraction < 1.0:
            cmd.extend([
                "--gpu_compute_fraction", str(config.gpu_compute_fraction),
            ])
        if config.physx_num_threads is not None:
            cmd.extend([
                "--physx_num_threads", str(config.physx_num_threads),
            ])
        
        # Isaac Sim display options
        cmd.extend([
            "--livestream", "0",
            "--headless",
        ])
        
        logger.info(f"Running nested simulation for iteration {iteration}")
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Check system resources before starting
        check_system_resources()
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            timeout=7200,  # 2 hours timeout (grasp optimization takes longer)
            # By omitting capture_output, stdout/stderr stream live to the terminal
        )
        logger.info(f"Nested simulation completed for iteration {iteration}")
        
        # Verify that scores.csv was created
        scores_file = os.path.join(iter_dir, "scores.csv")
        if os.path.exists(scores_file):
            logger.info(f"Successfully verified scores.csv exists at: {scores_file}")
        else:
            logger.warning(f"scores.csv not found at expected location: {scores_file}")
            # Check if it was saved elsewhere
            output_scores = os.path.join(output_dir, "scores.csv")
            if os.path.exists(output_scores):
                logger.info(f"Found scores.csv in output_dir, copying to iteration directory...")
                import shutil
                shutil.copy2(output_scores, scores_file)
                logger.info(f"Copied scores.csv to: {scores_file}")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Nested simulation failed for iteration {iteration} with return code {e.returncode}")
        logger.error(f"Command was: {' '.join(cmd)}")
        
        # Special handling for return code -9 (SIGKILL, often OOM)
        if e.returncode == -9:
            logger.error("=" * 80)
            logger.error("PROCESS WAS KILLED (SIGKILL, return code -9)")
            logger.error("This typically indicates:")
            logger.error("  1. Out of Memory (OOM) - the OS killed the process due to memory exhaustion")
            logger.error("  2. Manual kill signal (SIGKILL)")
            logger.error("  3. System resource limits exceeded")
            logger.error("=" * 80)
            logger.error("Checking system resources...")
            check_system_resources()
            logger.error("=" * 80)
            logger.error("Suggestions:")
            logger.error("  - Reduce n_grasp_iter or n_grasp_sample to use less memory")
            logger.error("  - Reduce the number of environments (fewer hands/tasks)")
            logger.error("  - Check system logs: dmesg | grep -i oom")
            logger.error("  - Monitor memory usage during simulation")
            logger.error("=" * 80)
        
        if e.stdout:
            stdout_tail = e.stdout[-2000:] if len(e.stdout) > 2000 else e.stdout
            logger.error(f"Simulation stdout:\n{stdout_tail}")
        if e.stderr:
            stderr_tail = e.stderr[-2000:] if len(e.stderr) > 2000 else e.stderr
            logger.error(f"Simulation stderr:\n{stderr_tail}")
        
        # Try to read any partial output files that might exist
        scores_file = os.path.join(iter_dir, "scores.csv")
        if os.path.exists(scores_file):
            logger.warning(f"Found partial scores.csv despite error - simulation may have partially completed")
        
        return False
    except subprocess.TimeoutExpired:
        logger.error(f"Nested simulation timed out for iteration {iteration} after 2 hours")
        logger.error("Consider reducing n_grasp_iter or n_grasp_sample to reduce runtime")
        return False
    except Exception as e:
        logger.error(f"Error running nested simulation: {e}", exc_info=True)
        return False


def calculate_placeholder_score(params: Dict[str, Any]) -> float:
    """
    Calculate a placeholder score based on parameter values.
    Used when simulation script is not available.
    
    Score = 1 / (sum of absolute values of all parameters + epsilon)
    This favors configurations with smaller parameter values when minimizing.
    
    Args:
        params: Dictionary of parameter values
        
    Returns:
        Placeholder score value
    """
    epsilon = 1e-6  # Small value to avoid division by zero
    param_sum = sum(abs(v) if isinstance(v, (int, float)) else 0.0 for v in params.values())
    score = 1.0 / (param_sum + epsilon)
    return score


def read_scores(iteration: int, hand_index: int, config: NestedOptimizationConfig) -> Optional[float]:
    """
    Read score from scores.csv file.
    
    The nested simulation script saves scores.csv directly in the iteration directory.
    
    Args:
        iteration: Current iteration number
        hand_index: Index of hand in current iteration
        config: Optimization configuration
        
    Returns:
        Score value or None if not found
    """
    iter_dir = os.path.join(config.optimization_root, f"iter_{iteration}")
    scores_file = os.path.join(iter_dir, "scores.csv")
    
    if not os.path.exists(scores_file):
        logger.warning(f"Scores file not found: {scores_file}")
        return None
    
    try:
        df = pd.read_csv(scores_file)
        
        # Try to find score by hand_index
        if 'hand_index' in df.columns:
            hand_scores = df[df['hand_index'] == hand_index]
            if len(hand_scores) > 0:
                score = hand_scores.iloc[0][config.score_column]
                return float(score)
        
        # If no hand_index column, assume row index corresponds to hand_index
        if hand_index < len(df):
            if config.score_column in df.columns:
                score = df.iloc[hand_index][config.score_column]
                return float(score)
            else:
                # Use first numeric column
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    score = df.iloc[hand_index][numeric_cols[0]]
                    return float(score)
        
        logger.warning(f"Could not find score for hand_index {hand_index} in {scores_file}")
        return None
        
    except Exception as e:
        logger.error(f"Error reading scores file: {e}")
        return None


def objective(trial: optuna.Trial, config: NestedOptimizationConfig, iteration: int, hand_index: int) -> float:
    """
    Optuna objective function for nested optimization.
    
    Note: This function is not used directly in the nested optimization loop.
    Instead, we use ask-and-tell interface for better control.
    
    Args:
        trial: Optuna trial object
        config: Optimization configuration
        iteration: Current iteration number
        hand_index: Index of hand in current iteration
        
    Returns:
        Score value to optimize
    """
    # Sample parameters
    params = define_search_space(trial, config)
    
    # Generate hand models
    try:
        generate_hand_models(params, iteration, hand_index, config)
    except Exception as e:
        logger.error(f"Error generating hand models: {e}")
        # Return a bad score if generation fails
        return float('inf') if config.direction == "minimize" else float('-inf')
    
    # Save parameters
    save_parameters(params, iteration, hand_index, config)
    
    # Run nested simulation (only once per iteration, not per hand)
    if hand_index == 0:
        if not run_nested_simulation(iteration, config):
            logger.warning(f"Nested simulation failed for iteration {iteration}, using default score")
            return float('inf') if config.direction == "minimize" else float('-inf')
    
    # Read score
    score = read_scores(iteration, hand_index, config)
    
    if score is None:
        logger.warning(f"Could not read score for iteration {iteration}, hand {hand_index}")
        # Return a bad score if reading fails
        return float('inf') if config.direction == "minimize" else float('-inf')
    
    return score

def plot_optimization_progress(optimization_root: str = "./hand_optimization"):
    """
    Plot the average and best scores across optimization iterations.
    
    Args:
        optimization_root: Root directory containing iter_* subdirectories
    """
    optimization_path = Path(optimization_root)
    
    # Find all iteration directories
    iter_dirs = sorted([d for d in optimization_path.glob("iter_*") if d.is_dir()])
    
    if not iter_dirs:
        print(f"No iteration directories found in {optimization_root}")
        return
    
    iterations = []
    avg_scores = []
    best_scores = []
    num_samples = []
    
    # Read scores from each iteration
    for iter_dir in iter_dirs:
        scores_file = iter_dir / "scores.csv"
        
        if not scores_file.exists():
            print(f"Warning: {scores_file} not found, skipping iteration {iter_dir.name}")
            continue
        
        try:
            df = pd.read_csv(scores_file)
            
            if 'score' not in df.columns:
                print(f"Warning: 'score' column not found in {scores_file}")
                continue
            
            # Extract iteration number
            iter_num = int(iter_dir.name.split('_')[1])
            iterations.append(iter_num)
            
            # Calculate statistics
            scores = df['score'].astype(float)
            avg_scores.append(scores.mean())
            best_scores.append(scores.max())
            num_samples.append(len(scores))
            
        except Exception as e:
            print(f"Error reading {scores_file}: {e}")
            continue
    
    if not iterations:
        print("No valid scores found to plot")
        return
    # sort the iterations and the scores by the iterations
    iterations, avg_scores, best_scores, num_samples = zip(*sorted(zip(iterations, avg_scores, best_scores, num_samples)))
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Optimization Summary")
    print("="*60)
    print(f"Total iterations: {len(iterations)}")
    print(f"Initial average score: {avg_scores[0]:.6f}")
    print(f"Final average score: {avg_scores[-1]:.6f}")
    print(f"Improvement in average: {avg_scores[-1] - avg_scores[0]:.6f} ({((avg_scores[-1] - avg_scores[0]) / avg_scores[0] * 100):.2f}%)")
    print(f"\nInitial best score: {best_scores[0]:.6f}")
    print(f"Final best score: {best_scores[-1]:.6f}")
    print(f"Overall best score: {max(best_scores):.6f}")
    print(f"Improvement in best: {max(best_scores) - best_scores[0]:.6f} ({((max(best_scores) - best_scores[0]) / best_scores[0] * 100):.2f}%)")
    print("="*60)
    
    if not HAS_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib is required for plotting but is not installed. "
            "Install it (e.g., `pip install matplotlib`) and re-run."
        )

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot: Average and Best scores over iterations
    ax.plot(iterations, avg_scores, 'o-', label='Average Score', linewidth=2, markersize=8)
    ax.plot(iterations, best_scores, 's-', label='Best Score', linewidth=2, markersize=8)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Optimization Progress: Average and Best Scores', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.5, max(iterations) + 0.5)

    # Save the plot (required). Avoid plt.show() to prevent headless hangs.
    output_file = optimization_path / "optimization_progress.png"
    try:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
    except Exception as e:
        raise RuntimeError(f"Failed to save optimization plot to {output_file}: {e}") from e
    finally:
        plt.close(fig)

    print(f"Plot saved to: {output_file}")

def run_nested_optimization(config: NestedOptimizationConfig):
    """
    Main nested optimization loop.
    
    Outer loop: Hand-level TPE optimization
    Inner loop: Grasp-level TPE optimization (handled by nested_multi_grasp_selector.py)
    
    Args:
        config: Nested optimization configuration
    """
    # Guard against overwriting a previous run
    if not config.resume and os.path.isdir(config.optimization_root):
        for i in range(config.n_hand_iter):
            scores_file = os.path.join(config.optimization_root, f"iter_{i}", "scores.csv")
            if os.path.exists(scores_file):
                raise RuntimeError(
                    f"optimization_root '{config.optimization_root}' already contains completed iterations. "
                    "Use --resume to continue an existing run, or choose a different directory."
                )
            break  # only need to check iter_0

    # Create optimization root directory
    os.makedirs(config.optimization_root, exist_ok=True)

    # Create results directory
    results_dir = os.path.join(config.optimization_root, "optimization_results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration for reproducibility
    config_file = os.path.join(results_dir, "config.json")
    with open(config_file, 'w') as f:
        config_dict = {
            'base_hand_dir': config.base_hand_dir,
            'optimization_root': config.optimization_root,
            'simulation_script': config.simulation_script,
            'n_hand_iter': config.n_hand_iter,
            'n_samples_per_iter': config.n_samples_per_iter,
            'n_explore_samples_per_iter': config.n_explore_samples_per_iter,
            'n_grasp_iter': config.n_grasp_iter,
            'n_grasp_sample': config.n_grasp_sample,
            'top_k': config.top_k,
            'tasks': config.tasks,
            'device': config.device,
            'gpu_memory_fraction': config.gpu_memory_fraction,
            'gpu_compute_fraction': config.gpu_compute_fraction,
            'physx_num_threads': config.physx_num_threads,
            'direction': config.direction,
            'max_finger_number': config.max_finger_number,
            'thumb_number': config.thumb_number,
            'fingertip_scale_y_range': list(config.fingertip_scale_y_range),
            'fingertip_scale_z_range': list(config.fingertip_scale_z_range),
            'finger_link_added_length_range': list(config.finger_link_added_length_range),
            'thumb_link_added_length_range': list(config.thumb_link_added_length_range),
            'finger_code_options': config.finger_code_options,
            'thumb_code_options': config.thumb_code_options,
            'finger_number_options': config.finger_number_options,
            'finger_index2drop_options': config.finger_index2drop_options,
            'palm_bump_intensity_ratio_range': list(config.palm_bump_intensity_ratio_range),
            'study_name': config.study_name,
            'n_jobs': config.n_jobs,
            'n_parallel_hands': config.n_parallel_hands,
            'score_column': config.score_column,
            'hand_tpe_n_startup_trials': config.hand_tpe_n_startup_trials,
            'hand_tpe_n_ei_candidates': config.hand_tpe_n_ei_candidates,
            'hand_tpe_multivariate': config.hand_tpe_multivariate,
            'hand_tpe_seed': config.hand_tpe_seed,
            'hand_tpe_gamma': config.hand_tpe_gamma,
            'grasp_tpe_n_startup_trials': config.grasp_tpe_n_startup_trials,
            'grasp_tpe_n_ei_candidates': config.grasp_tpe_n_ei_candidates,
            'grasp_tpe_multivariate': config.grasp_tpe_multivariate,
            'grasp_tpe_seed': config.grasp_tpe_seed,
            'grasp_tpe_gamma': config.grasp_tpe_gamma,
        }
        json.dump(config_dict, f, indent=2)
    
    # Create Optuna study with TPE sampler
    # Use ask-and-tell interface for better control over iteration batching
    # Derive hand-level TPE hyperparameters if not explicitly set
    hand_n_startup = (
        config.hand_tpe_n_startup_trials
        if config.hand_tpe_n_startup_trials is not None
        else max(10, 3 * config.n_samples_per_iter)
    )
    hand_n_ei = (
        config.hand_tpe_n_ei_candidates
        if config.hand_tpe_n_ei_candidates is not None
        else max(24, config.n_samples_per_iter)
    )
    hand_seed = config.hand_tpe_seed
    hand_multivariate = config.hand_tpe_multivariate

    hand_sampler = optuna.samplers.TPESampler(
        n_startup_trials=hand_n_startup,
        n_ei_candidates=hand_n_ei,
        multivariate=hand_multivariate,
        seed=hand_seed,
    )
    # Optional: override gamma (good/bad split). This uses Optuna's internal API.
    if config.hand_tpe_gamma is not None:
        try:
            gamma_value = float(config.hand_tpe_gamma)
            if not (0.0 < gamma_value <= 0.5):
                logger.warning(
                    f"hand_tpe_gamma={gamma_value} is outside (0, 0.5]; ignoring custom gamma."
                )
            else:
                # Constant gamma across trials. This may change with Optuna versions.
                hand_sampler._gamma = lambda _n_trials: gamma_value  # type: ignore[attr-defined]
                logger.info(f"Using custom hand-level TPE gamma={gamma_value}")
        except Exception as e:  # pragma: no cover - defensive
            logger.warning(f"Failed to set custom hand_tpe_gamma: {e}")

    study = optuna.create_study(
        study_name=config.study_name,
        direction=config.direction,
        sampler=hand_sampler,
        storage=f"sqlite:///{os.path.join(results_dir, 'study.db')}",
        load_if_exists=True,
    )

    # --- Round-robin structural parameter exploration ---
    # Structural params (finger_code, thumb_code, finger_number, finger_index2drop)
    # must be shared across all hands in one iteration (PhysX requires identical
    # kinematic trees in a batch).  We enumerate ALL combinations, test each once
    # during an exploration phase, then fix the best combo for the remaining
    # exploitation iterations.
    all_structural_combos: List[Dict[str, Any]] = [
        {
            'finger_code': fc,
            'thumb_code': tc,
            'finger_number': fn,
            'finger_index2drop': fd,
        }
        for fn, fc, tc, fd in itertools.product(
            config.finger_number_options,
            config.finger_code_options,
            config.thumb_code_options,
            config.finger_index2drop_options,
        )
    ]
    n_structural_combos = len(all_structural_combos)
    # Track average score per structural combo during exploration
    structural_scores: Dict[int, List[float]] = {i: [] for i in range(n_structural_combos)}
    best_structural_combo: Optional[Dict[str, Any]] = None

    explore_trials = n_structural_combos * config.n_explore_samples_per_iter
    exploit_trials = max(0, config.n_hand_iter - n_structural_combos) * config.n_samples_per_iter
    total_trials = explore_trials + exploit_trials
    logger.info(f"Starting nested optimization")
    logger.info(f"  Hand-level iterations: {config.n_hand_iter}")
    logger.info(f"  Hands per iteration: {config.n_samples_per_iter} (exploit), {config.n_explore_samples_per_iter} (explore)")
    logger.info(f"  Total hand trials: {total_trials}")
    logger.info(f"  Structural combos: {n_structural_combos} (explore first {n_structural_combos} iters with 2x hands, then exploit best)")
    logger.info(f"  Grasp iterations per hand: {config.n_grasp_iter}")
    logger.info(f"  Grasp samples per combo: {config.n_grasp_sample}")
    logger.info(f"  Tasks: {config.tasks}")
    logger.info(f"  Optimization direction: {config.direction}")

    # Track trial number globally
    trial_counter = {'count': 0}

    # Determine start iteration (resume support)
    if config.resume:
        start_iteration = 0
        for i in range(config.n_hand_iter):
            scores_file = os.path.join(config.optimization_root, f"iter_{i}", "scores.csv")
            if os.path.exists(scores_file):
                start_iteration = i + 1
            else:
                break
        trial_counter['count'] = start_iteration * config.n_samples_per_iter
        n_existing_trials = len(study.trials)
        logger.info(f"Resuming from iteration {start_iteration} ({start_iteration} completed iterations found)")
        logger.info(f"  Optuna study has {n_existing_trials} existing trials")

        # Reconstruct structural exploration scores from completed iterations
        structural_results_file = os.path.join(results_dir, "structural_exploration.json")
        if os.path.exists(structural_results_file):
            with open(structural_results_file, 'r') as f:
                saved = json.load(f)
            structural_scores = {int(k): v for k, v in saved['scores'].items()}
            if saved.get('best_combo') is not None:
                best_structural_combo = saved['best_combo']
            logger.info(f"Restored structural exploration scores from {structural_results_file}")
        else:
            # Reconstruct from per-iteration scores.csv files
            for i in range(min(start_iteration, n_structural_combos)):
                scores_file = os.path.join(config.optimization_root, f"iter_{i}", "scores.csv")
                if os.path.exists(scores_file):
                    try:
                        df = pd.read_csv(scores_file)
                        if config.score_column in df.columns:
                            structural_scores[i] = df[config.score_column].tolist()
                    except Exception as e:
                        logger.warning(f"Could not read scores for structural combo {i}: {e}")
            # If exploration is complete, pick best combo
            if start_iteration >= n_structural_combos and best_structural_combo is None:
                compare = max if config.direction == "maximize" else min
                empty_default = float('-inf') if config.direction == "maximize" else float('inf')
                best_combo_idx = compare(
                    structural_scores,
                    key=lambda ci: np.mean(structural_scores[ci]) if structural_scores[ci] else empty_default,
                )
                best_structural_combo = all_structural_combos[best_combo_idx]
                logger.info(f"Reconstructed best structural combo (idx={best_combo_idx}): {best_structural_combo}")
    else:
        start_iteration = 0

    # Run hand-level optimization iteration by iteration
    for iteration in range(start_iteration, config.n_hand_iter):
        logger.info(f"\n{'='*60}")
        logger.info(f"HAND-LEVEL ITERATION {iteration + 1}/{config.n_hand_iter}")
        logger.info(f"{'='*60}")
        
        # Create iteration directory
        iter_dir = os.path.join(config.optimization_root, f"iter_{iteration}")
        os.makedirs(iter_dir, exist_ok=True)
        
        # --- Structural parameters: round-robin explore, then exploit ---
        # All hands in this batch must share the same URDF topology so that
        # PhysX ArticulationView can batch them (identical kinematic trees).
        if iteration < n_structural_combos:
            # Exploration phase: double the hands per iteration for more reliable scores
            trials_this_iter = config.n_explore_samples_per_iter
            combo_idx = iteration
            iter_structural_params = all_structural_combos[combo_idx]
            logger.info(f"Structural EXPLORE [{combo_idx+1}/{n_structural_combos}]: {iter_structural_params}")
        else:
            # Exploitation phase: pick the best combo seen so far
            if best_structural_combo is None:
                # First time entering exploit — select best from exploration scores
                compare = max if config.direction == "maximize" else min
                empty_default = float('-inf') if config.direction == "maximize" else float('inf')
                best_combo_idx = compare(
                    structural_scores,
                    key=lambda i: np.mean(structural_scores[i]) if structural_scores[i] else empty_default,
                )
                best_structural_combo = all_structural_combos[best_combo_idx]
                logger.info(f"Exploration complete. Best structural combo (idx={best_combo_idx}): {best_structural_combo}")
                logger.info(f"  Mean score: {np.mean(structural_scores[best_combo_idx]):.6f}")
                # Log all combo scores for comparison
                for ci in range(n_structural_combos):
                    scores = structural_scores[ci]
                    mean_s = f"{np.mean(scores):.6f}" if scores else "N/A"
                    logger.info(f"  Combo {ci}: mean={mean_s}, scores={scores}, params={all_structural_combos[ci]}")
            trials_this_iter = config.n_samples_per_iter
            iter_structural_params = best_structural_combo
            logger.info(f"Structural EXPLOIT: {iter_structural_params}")

        logger.info(f"Generating {trials_this_iter} hands for iteration {iteration}")

        # Save structural params for this iteration (for reproducibility)
        structural_file = os.path.join(iter_dir, "structural_params.json")
        with open(structural_file, 'w') as f:
            json.dump(iter_structural_params, f, indent=2)

        # Generate all hands for this iteration first
        # First, sample all parameters using TPE
        trials_to_generate = []
        for hand_index in range(trials_this_iter):
            trial = study.ask()  # Get a new trial
            params = define_search_space(trial, config)
            # Inject shared structural params into every hand's params
            params.update(iter_structural_params)
            trials_to_generate.append((trial, params, hand_index))
        
        # Generate hands in parallel
        trials_data = []
        if config.n_parallel_hands > 1 and trials_this_iter > 1:
            # Parallel generation
            logger.info(f"Generating hands in parallel (max {config.n_parallel_hands} workers)")
            with ThreadPoolExecutor(max_workers=min(config.n_parallel_hands, trials_this_iter)) as executor:
                # Submit all tasks
                future_to_trial = {
                    executor.submit(_generate_single_hand, trial, params, iteration, hand_index, config, trial_counter): 
                    (trial, params, hand_index)
                    for trial, params, hand_index in trials_to_generate
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_trial):
                    trial, params, hand_index, error = future.result()
                    if error is None:
                        trials_data.append((trial, params, hand_index))
                    else:
                        raise RuntimeError(f"Hand generation failed for iteration {iteration}, hand {hand_index}: {error}") from error
        else:
            # Sequential generation (original approach)
            for trial, params, hand_index in trials_to_generate:
                result = _generate_single_hand(trial, params, iteration, hand_index, config, trial_counter)
                trial, params, hand_index, error = result
                if error is None:
                    trials_data.append((trial, params, hand_index))
                else:
                    raise RuntimeError(f"Hand generation failed for iteration {iteration}, hand {hand_index}: {error}") from error
        
        # Run nested simulation (grasp optimization inside Isaac Sim)
        iter_scores = []  # Collect scores for structural study
        if trials_data:
            if config.simulation_script:
                # Run nested simulation (grasp-level optimization happens inside)
                logger.info(f"Running nested simulation for iteration {iteration}")
                logger.info(f"  This will run {config.n_grasp_iter} grasp optimization iterations")
                logger.info(f"  Isaac Sim will stay loaded for all grasp iterations")

                sim_success = run_nested_simulation(iteration, config)
                if not sim_success:
                    raise RuntimeError(f"Nested simulation failed for iteration {iteration}")

                # Double-check that scores.csv exists before trying to read it
                iter_dir = os.path.join(config.optimization_root, f"iter_{iteration}")
                scores_file = os.path.join(iter_dir, "scores.csv")
                if not os.path.exists(scores_file):
                    raise RuntimeError(
                        f"scores.csv not found after simulation for iteration {iteration} at: {scores_file}. "
                        "The simulation did not complete successfully or did not save scores."
                    )

                # Read hand scores from CSV (created by nested_multi_grasp_selector.py)
                for idx, (trial, params, hand_index) in enumerate(trials_data):
                    score = read_scores(iteration, hand_index, config)

                    if score is None:
                        raise RuntimeError(f"Could not read score for iteration {iteration}, hand {hand_index}")

                    # Report the score to Optuna (continuous params study)
                    study.tell(trial, score)
                    iter_scores.append(score)
                    logger.info(f"Hand {hand_index}: score = {score:.6f}")
            else:
                # Use placeholder score function (for testing without simulation)
                logger.info(f"Using placeholder score function (no simulation script provided)")
                for idx, (trial, params, hand_index) in enumerate(trials_data):
                    score = calculate_placeholder_score(params)

                    # Report the score to Optuna (continuous params study)
                    study.tell(trial, score)
                    iter_scores.append(score)
                    logger.info(f"Hand {hand_index}: placeholder score = {score:.6f}")

        # Record scores for structural combo tracking
        if iter_scores and iteration < n_structural_combos:
            structural_scores[iteration].extend(iter_scores)
            mean_score = np.mean(iter_scores)
            logger.info(f"Structural combo {iteration} mean score: {mean_score:.6f}")

            # Save structural exploration progress incrementally (for resume)
            structural_results_file = os.path.join(results_dir, "structural_exploration.json")
            with open(structural_results_file, 'w') as f:
                json.dump({
                    'combos': all_structural_combos,
                    'scores': {str(k): v for k, v in structural_scores.items()},
                    'best_combo_idx': None,
                    'best_combo': None,
                }, f, indent=2)

        # Log best value so far
        if study.best_trial:
            logger.info(f"Best hand score so far: {study.best_value:.6f}")
        
        # Plot progress after each iteration
        plot_optimization_progress(config.optimization_root)

    # Save results — merge best continuous params with best structural params
    best_combined = dict(study.best_params)
    if best_structural_combo is not None:
        best_combined.update(best_structural_combo)

    best_params_file = os.path.join(results_dir, "best_params.json")
    with open(best_params_file, 'w') as f:
        json.dump(best_combined, f, indent=2)

    # Save structural exploration results
    structural_results_file = os.path.join(results_dir, "structural_exploration.json")
    with open(structural_results_file, 'w') as f:
        json.dump({
            'combos': all_structural_combos,
            'scores': {str(k): v for k, v in structural_scores.items()},
            'best_combo_idx': all_structural_combos.index(best_structural_combo) if best_structural_combo else None,
            'best_combo': best_structural_combo,
        }, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"NESTED OPTIMIZATION COMPLETED!")
    logger.info(f"{'='*60}")
    logger.info(f"Best hand score: {study.best_value:.6f}")
    logger.info(f"Best parameters saved to: {best_params_file}")

    # Print best parameters
    logger.info("Best hand parameters (continuous):")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    if best_structural_combo is not None:
        logger.info("Best structural parameters:")
        for key, value in best_structural_combo.items():
            logger.info(f"  {key}: {value}")

    return study


def main():
    """Command-line interface for the nested optimization script."""
    import argparse

    # Single source of truth for default values
    _DEFAULTS = NestedOptimizationConfig()

    parser = argparse.ArgumentParser(
        description="Nested TPE Optimization for Hand Generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Base directories
    parser.add_argument(
        "--base_hand_dir",
        type=str,
        default=_DEFAULTS.base_hand_dir,
        help="Base hand configuration directory"
    )
    parser.add_argument(
        "--optimization_root",
        type=str,
        default=f"/home/Amin/opt_results/nested_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        help="Root directory for optimization"
    )
    parser.add_argument(
        "--simulation_script",
        type=str,
        default=_DEFAULTS.simulation_script,
        help="Path to nested simulation script"
    )

    # Hand-level optimization parameters
    parser.add_argument(
        "--n_hand_iter",
        type=int,
        default=_DEFAULTS.n_hand_iter,
        help="Number of hand-level optimization iterations"
    )
    parser.add_argument(
        "--n_samples_per_iter",
        type=int,
        default=_DEFAULTS.n_samples_per_iter,
        help="Number of hand configurations per iteration (exploitation phase)"
    )
    parser.add_argument(
        "--n_explore_samples_per_iter",
        type=int,
        default=_DEFAULTS.n_explore_samples_per_iter,
        help="Number of hand configurations per iteration (structural exploration phase)"
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default=_DEFAULTS.study_name,
        help="Optuna study name"
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["minimize", "maximize"],
        default=_DEFAULTS.direction,
        help="Optimization direction"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=_DEFAULTS.n_jobs,
        help="Number of parallel jobs"
    )
    parser.add_argument(
        "--n_parallel_hands",
        type=int,
        default=_DEFAULTS.n_parallel_hands,
        help="Number of hands to generate in parallel per iteration"
    )
    parser.add_argument(
        "--score_column",
        type=str,
        default=_DEFAULTS.score_column,
        help="Column name in scores.csv"
    )
    # Hand-level TPE hyperparameters
    parser.add_argument(
        "--hand_tpe_n_startup_trials",
        type=int,
        default=_DEFAULTS.hand_tpe_n_startup_trials,
        help="Hand-level TPE n_startup_trials (default: 3 * n_samples_per_iter, min 10)"
    )
    parser.add_argument(
        "--hand_tpe_n_ei_candidates",
        type=int,
        default=_DEFAULTS.hand_tpe_n_ei_candidates,
        help="Hand-level TPE n_ei_candidates (default: max(24, n_samples_per_iter))"
    )
    parser.add_argument(
        "--hand_tpe_multivariate",
        action="store_true",
        help="Use multivariate TPE for hand-level optimization"
    )
    parser.add_argument(
        "--hand_tpe_seed",
        type=int,
        default=_DEFAULTS.hand_tpe_seed,
        help="Random seed for hand-level TPE sampler (None = random)"
    )
    parser.add_argument(
        "--hand_tpe_gamma",
        type=float,
        default=_DEFAULTS.hand_tpe_gamma,
        help="Optional hand-level TPE gamma (good/bad split fraction, 0 < gamma <= 0.5). Uses internal Optuna API."
    )

    # Grasp-level optimization parameters (passed to nested_multi_grasp_selector.py)
    parser.add_argument(
        "--n_grasp_iter",
        type=int,
        default=_DEFAULTS.n_grasp_iter,
        help="Number of grasp optimization iterations (inner loop)"
    )
    parser.add_argument(
        "--n_grasp_sample",
        type=int,
        default=_DEFAULTS.n_grasp_sample,
        help="Number of grasp samples per hand-object combo per iteration"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=_DEFAULTS.top_k,
        help="Top-K grasps to average for final grasp score"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=_DEFAULTS.tasks,
        help="Comma-separated list of object types for simulation (same as original script)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=_DEFAULTS.device,
        help="Device for simulation (e.g., cuda:0, cpu)"
    )
    parser.add_argument(
        "--gpu_memory_fraction",
        type=float,
        default=_DEFAULTS.gpu_memory_fraction,
        help="Fraction of GPU memory to use (0.0 to 1.0, e.g., 0.5 for 50%%)"
    )
    parser.add_argument(
        "--gpu_compute_fraction",
        type=float,
        default=_DEFAULTS.gpu_compute_fraction,
        help="Fraction of GPU compute to use (0.0 to 1.0, e.g., 0.5 for 50%%). Limits parallel environments proportionally."
    )
    parser.add_argument(
        "--physx_num_threads",
        type=int,
        default=_DEFAULTS.physx_num_threads,
        help="Number of PhysX CPU threads (None = auto). Lower values reduce CPU/GPU load."
    )
    # Grasp-level TPE hyperparameters (forwarded to nested_multi_grasp_selector.py)
    parser.add_argument(
        "--grasp_tpe_n_startup_trials",
        type=int,
        default=_DEFAULTS.grasp_tpe_n_startup_trials,
        help="Grasp-level TPE n_startup_trials (default: 3 * n_grasp_sample)"
    )
    parser.add_argument(
        "--grasp_tpe_n_ei_candidates",
        type=int,
        default=_DEFAULTS.grasp_tpe_n_ei_candidates,
        help="Grasp-level TPE n_ei_candidates (default: max(24, n_grasp_sample))"
    )
    parser.add_argument(
        "--grasp_tpe_multivariate",
        action="store_true",
        help="Use multivariate TPE for grasp-level optimization"
    )
    parser.add_argument(
        "--grasp_tpe_seed",
        type=int,
        default=_DEFAULTS.grasp_tpe_seed,
        help="Random seed base for grasp-level TPE sampler (None = random)"
    )
    parser.add_argument(
        "--grasp_tpe_gamma",
        type=float,
        default=_DEFAULTS.grasp_tpe_gamma,
        help="Optional grasp-level TPE gamma (good/bad split fraction, 0 < gamma <= 0.5). Uses internal Optuna API."
    )

    # Parameter ranges
    parser.add_argument(
        "--fingertip_scale_y_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(_DEFAULTS.fingertip_scale_y_range),
        help="Fingertip scale Y range (shared across all fingers & thumbs)"
    )
    parser.add_argument(
        "--fingertip_scale_z_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(_DEFAULTS.fingertip_scale_z_range),
        help="Fingertip scale Z range (shared across all fingers & thumbs)"
    )
    parser.add_argument(
        "--finger_link_added_length_range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(_DEFAULTS.finger_link_added_length_range),
        help="Link added length range for fingers"
    )
    parser.add_argument(
        "--thumb_link_added_length_range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(_DEFAULTS.thumb_link_added_length_range),
        help="Link added length range for thumbs"
    )
    parser.add_argument(
        "--finger_code_options",
        type=str,
        nargs="+",
        default=_DEFAULTS.finger_code_options,
        help="Finger structure code options (categorical)"
    )
    parser.add_argument(
        "--thumb_code_options",
        type=str,
        nargs="+",
        default=_DEFAULTS.thumb_code_options,
        help="Thumb structure code options (categorical)"
    )
    parser.add_argument(
        "--finger_number_options",
        type=int,
        nargs="+",
        default=_DEFAULTS.finger_number_options,
        help="Finger count options (categorical)"
    )
    parser.add_argument(
        "--finger_index2drop_options",
        type=int,
        nargs="+",
        default=_DEFAULTS.finger_index2drop_options,
        help="Finger index to drop options (only matters when finger_number < max)"
    )
    parser.add_argument(
        "--palm_bump_intensity_ratio_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(_DEFAULTS.palm_bump_intensity_ratio_range),
        help="Palm bump intensity ratio range per bump"
    )

    # Resume support
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous run. Requires --optimization_root pointing to the existing run directory."
    )

    args = parser.parse_args()

    # Handle optimization_root default vs resume logic
    if args.resume:
        if args.optimization_root == parser.get_default("optimization_root"):
            parser.error("--resume requires --optimization_root pointing to an existing run directory")
        if not os.path.isdir(args.optimization_root):
            parser.error(f"--resume: optimization_root directory does not exist: {args.optimization_root}")
        results_dir = os.path.join(args.optimization_root, "optimization_results")
        study_db = os.path.join(results_dir, "study.db")
        if not os.path.exists(study_db):
            parser.error(f"--resume: study.db not found in {results_dir}. Cannot resume without a study database.")

    # Create configuration
    config = NestedOptimizationConfig(
        base_hand_dir=args.base_hand_dir,
        optimization_root=args.optimization_root,
        simulation_script=args.simulation_script,
        n_hand_iter=args.n_hand_iter,
        n_samples_per_iter=args.n_samples_per_iter,
        n_explore_samples_per_iter=args.n_explore_samples_per_iter,
        study_name=args.study_name,
        direction=args.direction,
        n_jobs=args.n_jobs,
        n_parallel_hands=args.n_parallel_hands,
        score_column=args.score_column,
        hand_tpe_n_startup_trials=args.hand_tpe_n_startup_trials,
        hand_tpe_n_ei_candidates=args.hand_tpe_n_ei_candidates,
        hand_tpe_multivariate=args.hand_tpe_multivariate,
        hand_tpe_seed=args.hand_tpe_seed,
        hand_tpe_gamma=args.hand_tpe_gamma,
        n_grasp_iter=args.n_grasp_iter,
        n_grasp_sample=args.n_grasp_sample,
        top_k=args.top_k,
        tasks=args.tasks,
        device=args.device,
        gpu_memory_fraction=args.gpu_memory_fraction,
        gpu_compute_fraction=args.gpu_compute_fraction,
        physx_num_threads=args.physx_num_threads,
        grasp_tpe_n_startup_trials=args.grasp_tpe_n_startup_trials,
        grasp_tpe_n_ei_candidates=args.grasp_tpe_n_ei_candidates,
        grasp_tpe_multivariate=args.grasp_tpe_multivariate,
        grasp_tpe_seed=args.grasp_tpe_seed,
        grasp_tpe_gamma=args.grasp_tpe_gamma,
        resume=args.resume,
        fingertip_scale_y_range=tuple(args.fingertip_scale_y_range),
        fingertip_scale_z_range=tuple(args.fingertip_scale_z_range),
        finger_link_added_length_range=tuple(args.finger_link_added_length_range),
        thumb_link_added_length_range=tuple(args.thumb_link_added_length_range),
        finger_code_options=args.finger_code_options,
        thumb_code_options=args.thumb_code_options,
        finger_number_options=args.finger_number_options,
        finger_index2drop_options=args.finger_index2drop_options,
        palm_bump_intensity_ratio_range=tuple(args.palm_bump_intensity_ratio_range),
    )
    
    # Detect finger and thumb numbers from base hand
    try:
        base_palm_cfg = PalmConfig(data=os.path.join(config.base_hand_dir, "palm_cfg.py"))
        config.max_finger_number = base_palm_cfg.finger_number
        config.thumb_number = base_palm_cfg.thumb_number
        logger.info(f"Detected {config.max_finger_number} max fingers and {config.thumb_number} thumbs from base hand")
    except Exception as e:
        logger.warning(f"Could not detect finger/thumb numbers from base hand: {e}")
        logger.info("Using default values: 3 max fingers, 1 thumb")
    
    # Run nested optimization
    study = run_nested_optimization(config)
    
    return study


if __name__ == "__main__":
    main()
