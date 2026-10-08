# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Centralized path configuration for robotouch.

This module provides dynamic path resolution based on the project root,
eliminating hardcoded absolute paths.
"""

from pathlib import Path

# Project root is the parent of the robotouch package directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

# Main directories
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
URDFS_DIR = PROJECT_ROOT / "urdfs"

# Asset subdirectories
OBJECTS_DIR = ASSETS_DIR / "objects"
LEAP_HAND_DIR = ASSETS_DIR / "leap_hand" / "leap_hand_right"

# Common asset paths
LEAP_HAND_USD = LEAP_HAND_DIR / "leap_hand_right_no_base.usd"

# Special files
EGRASPS_PATH = PROJECT_ROOT / "egrasps.npy"


def get_object_usd_path(object_name: str) -> str:
    """Get USD path for an object."""
    return str(OBJECTS_DIR / f"{object_name}.usd")


def get_ghost_object_usd_path(object_name: str) -> str:
    """Get USD path for a ghost object."""
    return str(OBJECTS_DIR / f"ghost_{object_name}.usd")


def get_task_data_dir(task: str) -> Path:
    """Get data directory for a specific task."""
    return DATA_DIR / task


def get_trajectories_path(task: str) -> str:
    """Get trajectories path for a task."""
    return str(DATA_DIR / task / "trajectories")


def get_ee_trajectory_dir(task: str, grasp_selection: int) -> str:
    """Get end-effector trajectory output directory."""
    return str(DATA_DIR / task / "ee_trajectory" / str(grasp_selection))


def get_topK_info_path(object_name: str) -> str:
    """Get topK info checkpoint path."""
    return str(CHECKPOINTS_DIR / f"{object_name}_topK_info.pt")


def get_histogram_path(object_name: str) -> str:
    """Get histogram path."""
    return str(CHECKPOINTS_DIR / f"{object_name}_histogram.png")
