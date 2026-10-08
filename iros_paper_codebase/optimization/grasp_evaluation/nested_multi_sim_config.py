"""
Nested Multi-hand multi-task grasp selection environment configuration.

This environment runs multiple hands across multiple tasks (hammer, stir, knife, saw)
in parallel, testing grasp stability through 12 force/torque tests per grasp.

Modified from multi_sim_config.py to support:
- Per-environment grasp parameters (obj_dist_palm, obj_angle_palm, finger_spread, finger_m_rand)
- set_grasp_configs() method to set grasp parameters for each environment
- Always uses provided grasp configs from optimizer (no random sampling mode)
"""

from __future__ import annotations
from utils import load_transforms
import numpy as np
import math
import torch
from collections.abc import Sequence
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation, RigidObjectCfg, RigidObject, AssetBaseCfg, AssetBase
from isaaclab.envs import DirectRLEnvCfg, DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul, quat_inv, quat_apply

from omni.physx import get_physx_replicator_interface, get_physx_simulation_interface
from pxr import UsdUtils

from robotouch.hand_config import HandConfig, JointInfo
from robotouch.urdf_padder import PaddedHandInfo
from robotouch.paths import OBJECTS_DIR, get_object_usd_path, get_trajectories_path


# Task-specific constants (indexed by task)
TASK_NAMES = ["hammer", "stir", "knife", "saw"]
TASK_OBJECT_NAMES = ["hammer", "spoon", "knife", "saw"]

# Task parameters as tensors for vectorized lookup
# Shape: [4, 3] for positions, [4] for scalars
# TASK_EE_OFFSET_POS = torch.tensor([
#     [0.08, 0.07, 0.18],  # hammer
#     [0.08, 0.07, 0.20],  # stir
#     [0.08, 0.07, 0.26],  # knife
#     [0.10, 0.00, 0.41],  # saw
# ], dtype=torch.float32)


TASK_EE_OFFSET_POS = torch.tensor([
    [0.08, 0.065, 0.18],  # hammer
    [0.08, 0.058, 0.20],  # stir
    [0.08, 0.065, 0.26],  # knife
    [0.10, 0.00, 0.41],  # saw
], dtype=torch.float32)


TASK_EE_OFFSET_ANGLE = torch.tensor([
    [np.pi/2, np.pi, 0],  # hammer
    [np.pi/2, np.pi, 0],  # stir
    [np.pi/2, np.pi, 0],  # knife
    [0, np.pi, 0],        # saw
], dtype=torch.float32)

# Note: TASK_EE_RAND_POS and TASK_EE_RAND_ANGLE are not used in nested optimization mode
# All grasp parameters (obj_dist_palm, obj_angle_palm) come from the TPE optimizer via set_grasp_configs()

TASK_OBJECT_MASSES = torch.tensor([0.514, 0.060, 0.157, 0.385], dtype=torch.float32)

TASK_CENTROID_OFFSETS = torch.tensor([
    [0.0, 0.0, -0.15],   # hammer
    [0.0, 0.0, -0.163],  # stir
    [0.0, 0.0, -0.17],   # knife
    [0.0, 0.0, -0.205],  # saw
], dtype=torch.float32)


def _build_object_cfgs(base_path: str = None) -> Tuple[List[str], List[str]]:
    if base_path is None:
        base_path = str(OBJECTS_DIR)
    """Build object USD paths for each task.

    Returns paths instead of configs to avoid nested rigid body issues.
    Physics properties will be applied at the MultiAssetSpawnerCfg level.
    """
    object_paths = []
    ghost_paths = []

    for obj_name in TASK_OBJECT_NAMES:
        object_paths.append(f"{base_path}/{obj_name}.usd")
        ghost_paths.append(f"{base_path}/ghost_{obj_name}.usd")

    return object_paths, ghost_paths


def _get_revolute_joint_order(urdf_path: str) -> List[str]:
    """Get revolute joint names in URDF declaration order."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joint_names: List[str] = []
    for joint in root.findall('joint'):
        if joint.get('type') == 'revolute':
            name = joint.get('name')
            if name:
                joint_names.append(name)
    return joint_names


def build_multi_robot_cfg(
    padded_hands: List[PaddedHandInfo],
    joint_template: List[str],
    num_hands: int,
    num_tasks: int,
    fix_root_link: bool = True,
) -> ArticulationCfg:
    """
    Build ArticulationCfg for multiple hands using MultiUsdFileCfg.

    This function uses the same physics/actuator settings as robot_builder.build_robot_cfg.
    Key design notes:
    - USD paths are replicated for each task to match the environment layout
    - Initial joint positions are set to 0; per-hand positions (thumb fanning, etc.)
      are applied in _reset_idx using each hand's joint_info
    - Initial rotation is identity since it's overwritten every step in _apply_action
      via write_root_pose_to_sim using per-hand base_rotation_per_env
    - Base offset handling is done during URDF->USD conversion (see urdf_padder.py)

    Requires that padded_hands have their usd_path field populated.
    """
    # Build list of USD paths: each hand appears num_tasks times
    usd_paths = []
    for hand_info in padded_hands:
        if not hand_info.usd_path:
            raise ValueError(f"Hand {hand_info.hand_name} has no USD path. "
                           "Call convert_all_padded_hands_to_usd() first.")
        for _ in range(num_tasks):
            usd_paths.append(hand_info.usd_path)

    # Initial joint positions: all zeros (closing joints)
    # Per-hand positions (thumb fanning at ±π/2, axial joints) are set in _reset_idx
    # using the hand-specific joint_info from HandConfig
    joint_pos_dict = {}
    for joint_name in joint_template:
        joint_pos_dict[f"^({joint_name})$"] = 0.0

    # Use same physics settings as robot_builder.build_robot_cfg
    return ArticulationCfg(
        prim_path="/World/envs/env_.*/robot",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=usd_paths,
            random_choice=False,  # We control assignment via env index
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=1,
            ),
            activate_contact_sensors=False,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=1,
                fix_root_link=fix_root_link,
            ),
        ),
        actuators={
            # Same actuator settings as robot_builder.build_robot_cfg
            "hand": ImplicitActuatorCfg(
                joint_names_expr=joint_template,
                effort_limit_sim=2.0,
                stiffness=2.0,
                damping=0.3,
            ),
        },
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            # Identity rotation - actual per-hand rotation is applied every step
            # in _apply_action via write_root_pose_to_sim using base_rotation_per_env
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=joint_pos_dict,
        )
    )


@configclass
class MultiSimEnvCfg(DirectRLEnvCfg):
    """Configuration for multi-hand multi-task grasp selection environment."""

    debug = False

    # Multi-hand multi-task settings (set by prepare_config)
    # Using Any type to skip configclass validation for complex nested objects
    padded_hands: Any = None  # List[PaddedHandInfo] - set by prepare_config
    joint_template: Any = None  # List[str] - set by prepare_config
    num_hands: int = 0
    num_tasks: int = 4
    tasks: Any = None  # List[str] - set by prepare_config

    envs_per_combo: int = 0
    max_joints: int = 0
    k: int = 5  # Number of top grasps to track
    
    # Single-shot mode: each env runs exactly once (no reset), simulation ends when all envs complete
    single_shot_mode: bool = False

    def validate(self):
        """Override validation to skip problematic nested configs."""
        # Skip validation - our config has complex nested structures
        # that Isaac Lab's configclass validation can't handle
        pass

    # Timing and physics
    dt = 1/120
    decimation = 2
    episode_length_s = 30.0  # Required by DirectRLEnvCfg base class
    action_scale = 0.05  # Required by DirectRLEnvCfg base class
    action_space = 16  # Will be updated to max_joints
    observation_space = 117  # Required by DirectRLEnvCfg base class
    state_space = 0  # Required by DirectRLEnvCfg base class

    delay = 60.0
    min_post_close_buffer = 1.0

    transforms_window = 3
    transforms_sample = 180
    object_start_pos = [0.0, 0.0, 0.0]
    object_start_euler = [0.0, np.pi/2, 0.0]
    fixed_object_mass = 5000.0

    # Finger closing parameters
    joint_speed_fast = 0.1
    joint_speed_slow = 0.045
    thumb_wave_delay = 2  # Delay thumb closing by N waves (thumb starts after other fingers)
    torque_approach_threshold = 0.35
    palm_speed = 0.0015
    object_threshold = 0.0002
    palm_recoil = 0.7
    torque_min = 0.55
    torque_max = torque_min
    # Note: finger_spread is not used - replaced by grasp_finger_spread from TPE optimizer
    finger_group_delay = 0.05
    # Note: finger_actuate_min and finger_actuate_max are not used in nested optimization mode
    # finger_m_rand comes from the TPE optimizer via set_grasp_configs()

    # Test parameters
    time_per_test = 1
    test_delay = 0.03
    newton_per_test = 100
    torque_per_test = 10.0
    max_test_angle = (6 * math.pi / 180)
    max_test_pos = 0.008

    # Scene config - CRITICAL: replicate_physics=False for multi-asset
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048,
        env_spacing=0.6,
        replicate_physics=False,
    )

    sim: SimulationCfg = SimulationCfg(
        dt=1/120,
        render_interval=2,
        physx=PhysxCfg(
            # GPU buffer sizing: increase these when PhysX reports overflows / missing interactions.
            # Values are chosen as powers-of-two above the required capacities reported by PhysX.
            gpu_max_rigid_contact_count=2**25,  # >= 18,038,840 required (contact buffer overflow)
            gpu_max_rigid_patch_count=2**23,    # >= 5,389,017 required (patch buffer overflow)
            # Fix PhysX warning:
            # "increase PxGpuDynamicsMemoryConfig::foundLostPairsCapacity ..."
            # See: PhysxCfg.gpu_found_lost_pairs_capacity in IsaacLab.
            gpu_found_lost_pairs_capacity=2**23,  # 8,388,608 > 6,338,758 required
            # Optional headroom for dense aggregate scenes (AABB manager). Uncomment if you see aggregate warnings.
            # gpu_found_lost_aggregate_pairs_capacity=2**26,
        ),
        # Enable stage in memory mode - required for partial physics replication
        # This makes physics load via attach_stage() instead of force_load_physics_from_usd()
        create_stage_in_memory=True,
    )

    # Asset configs (set by prepare_config)
    robot_cfg: ArticulationCfg = None
    object_cfg: RigidObjectCfg = None
    swapped_object_cfg: RigidObjectCfg = None
    ghost_object_cfg: RigidObjectCfg = None

    def prepare_config(
        self,
        padded_hands: List[PaddedHandInfo],
        joint_template: List[str],
        tasks: List[str],
        k: int = 5,
    ):
        """
        Prepare configuration with hand and task information.

        Must be called before creating the environment.
        """
        self.padded_hands = padded_hands
        self.joint_template = joint_template
        self.num_hands = len(padded_hands)
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.k = k
        self.max_joints = len(joint_template)
        self.action_space = self.max_joints

        # Compute envs per combo
        num_combos = self.num_hands * self.num_tasks
        self.envs_per_combo = self.scene.num_envs // num_combos
        # Adjust num_envs to be divisible by num_combos
        self.scene.num_envs = self.envs_per_combo * num_combos

        # Build robot config
        self.robot_cfg = build_multi_robot_cfg(
            padded_hands,
            joint_template,
            self.num_hands,
            self.num_tasks,
            fix_root_link=True,
        )

        # Build object USD paths
        object_paths, ghost_paths = _build_object_cfgs()

        # Get task indices from task names
        task_indices = [TASK_NAMES.index(t) for t in tasks]
        selected_obj_paths = [object_paths[i] for i in task_indices]
        selected_ghost_paths = [ghost_paths[i] for i in task_indices]

        # Build per-task object configs
        selected_obj_cfgs = [
            sim_utils.UsdFileCfg(
                usd_path=path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=True,
                    max_depenetration_velocity=1,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.fixed_object_mass),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            )
            for path in selected_obj_paths
        ]
        selected_ghost_cfgs = [
            sim_utils.UsdFileCfg(
                usd_path=path,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            )
            for path in selected_ghost_paths
        ]

        # Replicate configs for each hand to match env layout
        replicated_obj_cfgs = []
        replicated_ghost_cfgs = []
        for _ in range(self.num_hands):
            replicated_obj_cfgs.extend(selected_obj_cfgs)
            replicated_ghost_cfgs.extend(selected_ghost_cfgs)

        # Fixed object config - use MultiAssetSpawnerCfg for per-env assets
        self.object_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/object",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=replicated_obj_cfgs,
                random_choice=False,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                lin_vel=[0.0, 0.0, 0.0],
                pos=[0.0, 0.0, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0],
            )
        )

        # Swapped object config - with per-task mass
        swapped_obj_cfgs = []
        for task_name in tasks:
            task_idx = TASK_NAMES.index(task_name)
            obj_name = TASK_OBJECT_NAMES[task_idx]
            mass = TASK_OBJECT_MASSES[task_idx].item()
            swapped_obj_cfgs.append(
                sim_utils.UsdFileCfg(
                    usd_path=get_object_usd_path(obj_name),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        disable_gravity=True,
                        max_depenetration_velocity=1,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                )
            )

        replicated_swapped_cfgs = []
        for _ in range(self.num_hands):
            replicated_swapped_cfgs.extend(swapped_obj_cfgs)

        self.swapped_object_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/swapped_object",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=replicated_swapped_cfgs,
                random_choice=False,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                lin_vel=[0.0, 0.0, 0.0],
                pos=[0.0, 0.0, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0],
            )
        )

        # Ghost object config - per-env assets
        self.ghost_object_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/ghost_object",
            spawn=sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=replicated_ghost_cfgs,
                random_choice=False,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                lin_vel=[0.0, 0.0, 0.0],
                pos=[0.0, 0.0, 0.0],
                rot=[1.0, 0.0, 0.0, 0.0],
            )
        )


class NestedMultiSimEnv(DirectRLEnv):
    """
    Nested Multi-hand multi-task grasp selection environment.

    Runs multiple hands across multiple tasks in parallel, testing grasp
    stability through 12 force/torque tests per grasp.
    
    Modified to support per-environment grasp parameters for nested optimization.
    """

    cfg: MultiSimEnvCfg

    def __init__(self, cfg: MultiSimEnvCfg, render_mode: str | None = None, **kwargs):
        # Monkey-patch validate to skip validation for complex nested configs
        # This is necessary because Isaac Lab's configclass can't handle
        # MultiAssetSpawnerCfg with list of configs
        original_validate = cfg.validate
        cfg.validate = lambda: None

        import time
        print(f"[TIMING] Starting scene setup with {cfg.scene.num_envs} envs...")
        print(f"[TIMING] Note: replicate_physics=False means each env gets its own physics (slow)")
        t0 = time.time()
        super().__init__(cfg, render_mode, **kwargs)
        print(f"[TIMING] Scene setup completed in {time.time() - t0:.1f}s")

        # Restore original validate (in case it's needed later)
        cfg.validate = original_validate

        # === Environment Index Mapping (VECTORIZED - no for loops) ===
        env_indices = torch.arange(self.scene.num_envs, device=self.device)
        combo_indices = env_indices % (cfg.num_hands * cfg.num_tasks)
        self.env_to_hand = combo_indices // cfg.num_tasks  # [num_envs]
        self.env_to_task = combo_indices % cfg.num_tasks   # [num_envs]
        self.combo_indices = combo_indices  # [num_envs]
        self.num_combos = cfg.num_hands * cfg.num_tasks

        # Move task parameters to device for vectorized lookup
        self.task_ee_offset_pos = TASK_EE_OFFSET_POS.to(self.device)
        self.task_ee_offset_angle = TASK_EE_OFFSET_ANGLE.to(self.device)
        # Note: TASK_EE_RAND_POS and TASK_EE_RAND_ANGLE are not used in nested optimization mode
        self.task_centroid_offsets = TASK_CENTROID_OFFSETS.to(self.device)
        self.task_object_masses = TASK_OBJECT_MASSES.to(self.device)

        # Map task names to indices
        self.task_name_to_idx = {name: i for i, name in enumerate(TASK_NAMES)}
        self.task_indices = torch.tensor(
            [self.task_name_to_idx[t] for t in cfg.tasks],
            device=self.device
        )

        # Per-env task parameters (vectorized lookup)
        local_task_idx = self.env_to_task  # Index into cfg.tasks (0 to num_tasks-1)
        global_task_idx = self.task_indices[local_task_idx]  # Index into TASK_NAMES

        self.ee_offset_pos_base = self.task_ee_offset_pos[global_task_idx]  # [num_envs, 3]
        self.ee_offset_angle_base = self.task_ee_offset_angle[global_task_idx]  # [num_envs, 3]
        # Note: TASK_EE_RAND_POS and TASK_EE_RAND_ANGLE are not used in nested optimization mode
        # All grasp parameters come from the TPE optimizer
        self.obj_dist = -self.task_centroid_offsets[global_task_idx]  # [num_envs, 3]

        # Build real joint masks per hand as tensors
        self.real_joint_masks = []
        for hand_info in cfg.padded_hands:
            mask = torch.tensor(hand_info.real_joint_mask, device=self.device)
            self.real_joint_masks.append(mask)

        # Per-env real joint mask (vectorized)
        self.real_joint_mask_tensor = torch.stack(self.real_joint_masks, dim=0)  # [num_hands, max_joints]
        self.real_joint_mask_per_env = self.real_joint_mask_tensor[self.env_to_hand]  # [num_envs, max_joints]

        # Build hand configs lookup
        self.hand_configs = [h.hand_config for h in cfg.padded_hands]

        # Build base rotations per hand
        self.base_rotations = []
        for hand_info in cfg.padded_hands:
            rot = torch.tensor(hand_info.hand_config.base_rotation, device=self.device)
            self.base_rotations.append(rot)

        # Per-env base rotation
        self.base_rotations_tensor = torch.stack(self.base_rotations, dim=0)  # [num_hands, 4]
        self.base_rotation_per_env = self.base_rotations_tensor[self.env_to_hand]  # [num_envs, 4]

        # Number of joints
        self.num_joints = cfg.max_joints

        # Load transforms for each task
        self.transforms_per_task = {}
        for task_name in cfg.tasks:
            transforms_path = get_trajectories_path(task_name)
            task_idx = TASK_NAMES.index(task_name)
            centroid_offset = TASK_CENTROID_OFFSETS[task_idx].numpy()

            transforms_raw = load_transforms(
                transforms_path,
                cfg.object_start_pos,
                cfg.object_start_euler,
                cfg.transforms_window,
                cfg.transforms_sample
            )
            
            # Handle both list and tensor returns from load_transforms
            if isinstance(transforms_raw, list):
                if len(transforms_raw) == 0:
                    raise ValueError(f"No transforms found for task {task_name} at {transforms_path}")
                # Stack list of tensors into a single tensor
                transforms = torch.stack(transforms_raw).to(self.device)
            else:
                # Already a tensor
                transforms = transforms_raw.to(self.device)

            # Apply centroid offset
            transforms[:, :3] += quat_apply(
                transforms[:, 3:7],
                torch.tensor(centroid_offset, device=self.device).unsqueeze(0).expand(transforms.shape[0], -1)
            )
            self.transforms_per_task[task_name] = transforms

        # Precompute initial transforms per task and per env (vectorized)
        self.task_initial_transforms = torch.stack(
            [self.transforms_per_task[task_name][0] for task_name in cfg.tasks],
            dim=0
        )  # [num_tasks, 7]
        self.initial_transforms_per_env = self.task_initial_transforms[self.env_to_task]  # [num_envs, 7]

        # Initialize state tensors
        self.target_hand_pos = self.robot.data.default_joint_pos[:, :].clone()
        self.target_ee_pos_b = torch.zeros((self.scene.num_envs, 3), device=self.device)
        self.target_ee_quat_b = self.base_rotation_per_env.clone()

        self.ee_pos_offset = self.ee_offset_pos_base.clone()
        self.ee_angle_offset = self.ee_offset_angle_base.clone()

        # Defer object_init_pos initialization - will be set after first sim step
        # With replicate_physics=False, data isn't available until simulation runs
        self.object_init_pos = None
        self._object_init_pos_initialized = False

        self.timestep = torch.zeros(self.scene.num_envs, dtype=torch.int, device=self.device)
        self.delayed = torch.ones(self.scene.num_envs, dtype=torch.int, device=self.device) * \
                       int(cfg.delay / (cfg.sim.dt * cfg.decimation))

        self.finger_wait = torch.ones(self.scene.num_envs, dtype=torch.int, device=self.device) * \
                          int(cfg.finger_group_delay / cfg.sim.dt)

        # Build per-hand articulation order mapping
        # CRITICAL: Use self.robot.joint_names as ground truth (actual Isaac Lab articulation order)
        # This ensures finger_order, closing_directions, etc. match the actual physics simulation
        actual_art_names = list(self.robot.joint_names)

        if cfg.debug:
            print(f"=== ARTICULATION JOINT ORDER (ground truth from Isaac Lab) ===")
            for i, name in enumerate(actual_art_names):
                print(f"  {i}: {name}")

        self.hand_articulation_names = []
        self.hand_name_to_art_idx = []
        for hand_info in cfg.padded_hands:
            # Note: URDF order often differs from articulation order because PhysX
            # reorders joints via breadth-first kinematic tree traversal.
            # This is normal - we use the articulation order (ground truth).
            self.hand_articulation_names.append(actual_art_names)
            self.hand_name_to_art_idx.append({name: idx for idx, name in enumerate(actual_art_names)})

        # Build finger_order per hand using each hand's actual wave structure
        # Each hand has its own finger_order computed in hand_config based on:
        # - Wave N contains the Nth closing joint from each finger (synchronized closing)
        # - Joints are grouped by their ORDER within each finger, not absolute position
        # This preserves per-hand kinematic differences
        self.finger_orders = []

        for hand_idx, hand_info in enumerate(cfg.padded_hands):
            hand_config = hand_info.hand_config
            original_joint_names = hand_config.joint_names  # Original order
            name_to_art_idx = self.hand_name_to_art_idx[hand_idx]
            art_names = self.hand_articulation_names[hand_idx]

            # Map original finger_order indices to padded template indices
            # hand_config.finger_order is List[List[int]] where ints are indices into original_joint_names
            padded_finger_order = []
            for wave in hand_config.finger_order:
                padded_wave = []
                for orig_idx in wave:
                    orig_name = original_joint_names[orig_idx]
                    art_idx = name_to_art_idx.get(orig_name)
                    if art_idx is not None:
                        padded_wave.append(art_idx)
                if padded_wave:
                    padded_finger_order.append(padded_wave)

            self.finger_orders.append(padded_finger_order)

            if cfg.debug:
                print(f"  {hand_info.hand_name} finger_order ({len(padded_finger_order)} waves):")
                for i, wave in enumerate(padded_finger_order):
                    wave_names = [art_names[idx] for idx in wave]
                    print(f"    Wave {i}: {wave_names}")

        # Build fanning/axial joint mappings per hand
        # Fanning: first sideways joint per finger (base spreading)
        # Axial: remaining sideways joints (rotation)
        self.fanning_indices_per_hand = []  # List of lists of padded indices
        self.axial_indices_per_hand = []    # List of lists of padded indices
        self.joint_info_per_hand = []       # List of dicts mapping padded_idx -> JointInfo
        self.joints_by_finger_per_hand = [] # List of dicts mapping finger_id -> [padded_indices]

        for hand_idx, hand_info in enumerate(cfg.padded_hands):
            hand_config = hand_info.hand_config
            original_joint_names = hand_config.joint_names
            name_to_art_idx = self.hand_name_to_art_idx[hand_idx]

            # Map fanning indices to padded template
            padded_fanning = []
            for orig_idx in hand_config.fanning_joint_indices:
                orig_name = original_joint_names[orig_idx]
                art_idx = name_to_art_idx.get(orig_name)
                if art_idx is not None:
                    padded_fanning.append(art_idx)
            self.fanning_indices_per_hand.append(padded_fanning)

            # Map axial indices to padded template
            padded_axial = []
            for orig_idx in hand_config.axial_joint_indices:
                orig_name = original_joint_names[orig_idx]
                art_idx = name_to_art_idx.get(orig_name)
                if art_idx is not None:
                    padded_axial.append(art_idx)
            self.axial_indices_per_hand.append(padded_axial)

            # Build padded_idx -> JointInfo mapping for this hand
            padded_joint_info = {}
            for orig_name, info in hand_config.joint_info.items():
                art_idx = name_to_art_idx.get(orig_name)
                if art_idx is not None:
                    padded_joint_info[art_idx] = info
            self.joint_info_per_hand.append(padded_joint_info)

            # Build joints_by_finger mapping (finger_id -> [padded_indices]) for this hand
            joints_by_finger = {}
            for orig_name, info in hand_config.joint_info.items():
                art_idx = name_to_art_idx.get(orig_name)
                if art_idx is not None:
                    if info.finger_id not in joints_by_finger:
                        joints_by_finger[info.finger_id] = []
                    joints_by_finger[info.finger_id].append(art_idx)
            self.joints_by_finger_per_hand.append(joints_by_finger)

            if cfg.debug:
                art_names = self.hand_articulation_names[hand_idx]
                print(f"  {hand_info.hand_name} fanning: {[art_names[i] for i in padded_fanning]}")
                print(f"  {hand_info.hand_name} axial: {[art_names[i] for i in padded_axial]}")

        # Maximum waves across all hands (accounting for thumb delay)
        base_max_waves = max(len(fo) for fo in self.finger_orders) if self.finger_orders else 1
        max_waves = base_max_waves + cfg.thumb_wave_delay  # Extra waves for delayed thumb
        self.max_waves = max_waves

        # === PRE-COMPUTE PER-HAND TENSORS (loop over hands only, not envs) ===

        # 1. finger_groups_per_hand: [num_hands, max_waves, max_joints]
        # Thumb joints are delayed by thumb_wave_delay waves
        finger_groups_per_hand = torch.zeros(
            (cfg.num_hands, max_waves, self.num_joints),
            dtype=torch.int,
            device=self.device
        )
        for hand_idx, finger_order in enumerate(self.finger_orders):
            joint_info = self.joint_info_per_hand[hand_idx]
            for wave_idx, group in enumerate(finger_order):
                for joint_idx in group:
                    # Check if this joint is a thumb joint
                    info = joint_info.get(joint_idx)
                    if info is not None and info.finger_id == 't1':
                        # Delay thumb by thumb_wave_delay waves
                        delayed_wave = wave_idx + cfg.thumb_wave_delay
                        if delayed_wave < max_waves:
                            finger_groups_per_hand[hand_idx, delayed_wave, joint_idx] = 1
                    else:
                        # Non-thumb: use original wave
                        finger_groups_per_hand[hand_idx, wave_idx, joint_idx] = 1

        if cfg.debug:
            print(f"=== FINGER GROUPS WITH THUMB DELAY ({cfg.thumb_wave_delay} waves) ===")
            for hand_idx, hand_info in enumerate(cfg.padded_hands):
                art_names = self.hand_articulation_names[hand_idx]
                print(f"  {hand_info.hand_name}:")
                for wave_idx in range(max_waves):
                    active = finger_groups_per_hand[hand_idx, wave_idx].nonzero(as_tuple=True)[0].tolist()
                    if active:
                        names = [art_names[i] for i in active]
                        print(f"    Wave {wave_idx}: {names}")

        # 2. closing_directions_per_hand: [num_hands, max_joints]
        closing_directions_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints),
            dtype=torch.float,
            device=self.device
        )
        for hand_idx, hand_info in enumerate(cfg.padded_hands):
            name_to_art_idx = self.hand_name_to_art_idx[hand_idx]
            matched_count = 0
            unmatched_joints = []
            for orig_name, info in hand_info.hand_config.joint_info.items():
                art_idx = name_to_art_idx.get(orig_name)
                if art_idx is not None:
                    closing_directions_per_hand[hand_idx, art_idx] = info.closing_direction
                    matched_count += 1
                else:
                    unmatched_joints.append(orig_name)

            if cfg.debug or unmatched_joints:
                print(f"  {hand_info.hand_name} closing_directions: matched {matched_count}/{len(hand_info.hand_config.joint_info)} joints")
                if unmatched_joints:
                    print(f"    WARNING: Unmatched joints: {unmatched_joints}")

        # 3. non_thumb_joint_mask_per_hand: [num_hands, max_joints]
        non_thumb_joint_mask_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints),
            dtype=torch.bool,
            device=self.device
        )
        for hand_idx, joints_by_finger in enumerate(self.joints_by_finger_per_hand):
            for finger_id, joint_indices in joints_by_finger.items():
                if finger_id != 't1':
                    for idx in joint_indices:
                        non_thumb_joint_mask_per_hand[hand_idx, idx] = True

        # 4. Joint limits per hand: [num_hands, max_joints]
        joint_lower_limits_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints), dtype=torch.float, device=self.device
        )
        joint_upper_limits_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints), dtype=torch.float, device=self.device
        )
        for hand_idx, joint_info in enumerate(self.joint_info_per_hand):
            for padded_idx, info in joint_info.items():
                joint_lower_limits_per_hand[hand_idx, padded_idx] = info.lower_limit
                joint_upper_limits_per_hand[hand_idx, padded_idx] = info.upper_limit

        # 5. Fanning joint tensors per hand
        # fanning_mask: [num_hands, max_joints] - True for non-thumb fanning joints
        # fanning_fan_factors: [num_hands, max_joints] - fan factor for each (-1 to +1 range)
        # thumb_fanning_mask: [num_hands, max_joints] - True for thumb fanning joints
        # thumb_base_angles: [num_hands, max_joints] - base angle (π/2 or -π/2)
        fanning_mask_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints), dtype=torch.bool, device=self.device
        )
        fanning_fan_factors_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints), dtype=torch.float, device=self.device
        )
        thumb_fanning_mask_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints), dtype=torch.bool, device=self.device
        )
        thumb_base_angles_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints), dtype=torch.float, device=self.device
        )

        for hand_idx, (fanning_indices, joint_info) in enumerate(
            zip(self.fanning_indices_per_hand, self.joint_info_per_hand)
        ):
            # Separate thumb and non-thumb fanning
            non_thumb = [idx for idx in fanning_indices if joint_info[idx].finger_id != 't1']
            thumb = [idx for idx in fanning_indices if joint_info[idx].finger_id == 't1']

            # Sort non-thumb by finger order
            def _sort_key(idx):
                info = joint_info[idx]
                fid = info.finger_id
                if fid.startswith('f'):
                    try:
                        return (0, int(fid[1:]), info.position, idx)
                    except ValueError:
                        return (0, 999, info.position, idx)
                return (2, fid, info.position, idx)

            non_thumb_sorted = sorted(non_thumb, key=_sort_key)
            n_fingers = len(non_thumb_sorted)

            for j, idx in enumerate(non_thumb_sorted):
                fanning_mask_per_hand[hand_idx, idx] = True
                # Compute fan factor: +1, 0, -1 pattern for spreading
                if n_fingers >= 3:
                    fan_factor = 1.0 - (2.0 * j / (n_fingers - 1))
                elif n_fingers == 2:
                    fan_factor = 1.0 if j == 0 else -1.0
                else:
                    fan_factor = 0.0
                fanning_fan_factors_per_hand[hand_idx, idx] = fan_factor

            for idx in thumb:
                thumb_fanning_mask_per_hand[hand_idx, idx] = True
                info = joint_info[idx]
                base = np.pi / 2 if info.upper_limit >= np.pi / 2 else -np.pi / 2
                thumb_base_angles_per_hand[hand_idx, idx] = base

        # 6. Axial joint tensors per hand
        # axial_mask: [num_hands, max_joints] - True for axial joints
        # axial_alt_signs: [num_hands, max_joints] - alternating +1/-1 for pattern
        axial_mask_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints), dtype=torch.bool, device=self.device
        )
        axial_alt_signs_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints), dtype=torch.float, device=self.device
        )

        for hand_idx, axial_indices in enumerate(self.axial_indices_per_hand):
            for j, idx in enumerate(axial_indices):
                axial_mask_per_hand[hand_idx, idx] = True
                axial_alt_signs_per_hand[hand_idx, idx] = 1.0 if j % 2 == 0 else -1.0

        # 7. Thumb first closing joint (from robot_builder: starts at -1.2)
        # thumb_first_closing_mask: [num_hands, max_joints] - True for thumb first closing joint
        # thumb_first_closing_pos: [num_hands, max_joints] - -1.2 clamped to limits
        thumb_first_closing_mask_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints), dtype=torch.bool, device=self.device
        )
        thumb_first_closing_pos_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints), dtype=torch.float, device=self.device
        )

        for hand_idx, joint_info in enumerate(self.joint_info_per_hand):
            name_to_art_idx = self.hand_name_to_art_idx[hand_idx]
            # Find thumb closing joints and their positions
            thumb_closing = [(idx, info) for idx, info in joint_info.items()
                            if info.finger_id == 't1' and info.is_closing]
            if thumb_closing:
                # Find the first closing joint (smallest position value)
                first_pos = min(info.position for _, info in thumb_closing)
                for idx, info in thumb_closing:
                    if info.position == first_pos:
                        thumb_first_closing_mask_per_hand[hand_idx, idx] = True
                        # Target -1.2, clamped to limits (from robot_builder)
                        target = max(-1.2, info.lower_limit + 0.001)
                        target = min(target, info.upper_limit - 0.001)
                        thumb_first_closing_pos_per_hand[hand_idx, idx] = target

        # 8. Thumb axial joints (from robot_builder: start at midpoint of limits)
        # thumb_axial_mask: [num_hands, max_joints] - True for thumb axial (non-fanning sideways)
        # thumb_axial_midpoints: [num_hands, max_joints] - midpoint of limits
        thumb_axial_mask_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints), dtype=torch.bool, device=self.device
        )
        thumb_axial_midpoints_per_hand = torch.zeros(
            (cfg.num_hands, self.num_joints), dtype=torch.float, device=self.device
        )

        for hand_idx, joint_info in enumerate(self.joint_info_per_hand):
            fanning_set = set(self.fanning_indices_per_hand[hand_idx])
            for idx, info in joint_info.items():
                # Thumb sideways joints that are NOT the fanning joint (position 1)
                if info.finger_id == 't1' and not info.is_closing and idx not in fanning_set:
                    thumb_axial_mask_per_hand[hand_idx, idx] = True
                    mid = (info.lower_limit + info.upper_limit) / 2
                    thumb_axial_midpoints_per_hand[hand_idx, idx] = mid

        # Store per-hand tensors for use in _reset_idx
        self.finger_groups_per_hand = finger_groups_per_hand
        self.closing_directions_per_hand = closing_directions_per_hand
        self.non_thumb_joint_mask_per_hand = non_thumb_joint_mask_per_hand
        self.joint_lower_limits_per_hand = joint_lower_limits_per_hand
        self.joint_upper_limits_per_hand = joint_upper_limits_per_hand
        self.fanning_mask_per_hand = fanning_mask_per_hand
        self.fanning_fan_factors_per_hand = fanning_fan_factors_per_hand
        self.thumb_fanning_mask_per_hand = thumb_fanning_mask_per_hand
        self.thumb_base_angles_per_hand = thumb_base_angles_per_hand
        self.axial_mask_per_hand = axial_mask_per_hand
        self.axial_alt_signs_per_hand = axial_alt_signs_per_hand
        self.thumb_first_closing_mask_per_hand = thumb_first_closing_mask_per_hand
        self.thumb_first_closing_pos_per_hand = thumb_first_closing_pos_per_hand
        self.thumb_axial_mask_per_hand = thumb_axial_mask_per_hand
        self.thumb_axial_midpoints_per_hand = thumb_axial_midpoints_per_hand

        # === VECTORIZED EXPANSION TO ALL ENVS (no loops over envs) ===

        # finger_groups: [num_envs, max_waves, max_joints] - indexed by env_to_hand
        self.finger_groups = finger_groups_per_hand[self.env_to_hand].clone()

        self.current_finger_groups = torch.zeros(self.scene.num_envs, dtype=torch.int, device=self.device) - 1

        # closing_directions: [num_envs, max_joints] - indexed by env_to_hand
        self.closing_directions = closing_directions_per_hand[self.env_to_hand]

        # Torque stop thresholds
        random_torques = torch.rand(self.scene.num_envs, device=self.device) * \
                        (cfg.torque_max - cfg.torque_min) + cfg.torque_min
        self.torque_stop = random_torques.unsqueeze(1).repeat(1, self.num_joints)

        # Test state
        self.test_score = torch.zeros((self.scene.num_envs, 12), dtype=torch.int, device=self.device)
        self.test_cur = torch.zeros(self.scene.num_envs, dtype=torch.int, device=self.device)
        self.test_start = torch.zeros(self.scene.num_envs, dtype=torch.int, device=self.device)

        # Force/torque tests
        self.tests = torch.zeros((12, 6), dtype=torch.float, device=self.device)
        for i in range(6):
            self.tests[i, i // 2] = cfg.newton_per_test if i % 2 == 0 else -cfg.newton_per_test
        for i in range(6, 12):
            self.tests[i, i // 2] = cfg.torque_per_test if i % 2 == 0 else -cfg.torque_per_test

        # Initialize palm offset state (will be set from optimizer via set_grasp_configs)
        self.obj_dist_palm = torch.zeros((self.scene.num_envs, 3), device=self.device)
        self.obj_angle_palm = torch.zeros((self.scene.num_envs, 3), device=self.device)

        self.swap_add = torch.tensor([0., 0., 5., 0., 0., 0., 0.], device=self.device).unsqueeze(0).expand(
            self.scene.num_envs, -1).clone()

        # Pre-allocate reusable tensors
        self._zero_vel_full = torch.zeros((self.scene.num_envs, 6), device=self.device)
        self._zero_joint_vel = torch.zeros((self.scene.num_envs, self.num_joints), device=self.device)
        self.zeros = torch.zeros(self.scene.num_envs, dtype=torch.int, device=self.device)

        # Optimal state storage (for grasp data)
        self.opt_robot_joint_pos = self.robot.data.joint_pos.clone()
        self.opt_robot_target_pos = self.target_hand_pos.clone()
        self.opt_object_pos = torch.zeros((self.scene.num_envs, 3), device=self.device)
        self.opt_object_quat = torch.zeros((self.scene.num_envs, 4), device=self.device)
        self.opt_ee_offset = self.ee_pos_offset.clone()
        self.opt_ee_angle = self.ee_angle_offset.clone()
        self.opt_obj_dist_palm = self.obj_dist_palm.clone()
        self.opt_obj_angle_palm = self.obj_angle_palm.clone()

        # === TopK Tracking Per Combo (VECTORIZED) ===
        self.topK_scores = torch.full(
            (self.num_combos, cfg.k), -float("inf"), device=self.device
        )
        self.topK_joint_pos = torch.zeros(
            (self.num_combos, cfg.k, self.num_joints), device=self.device
        )
        self.topK_target_pos = torch.zeros(
            (self.num_combos, cfg.k, self.num_joints), device=self.device
        )
        self.topK_ee_pos = torch.zeros(
            (self.num_combos, cfg.k, 3), device=self.device
        )
        self.topK_ee_quat = torch.zeros(
            (self.num_combos, cfg.k, 4), device=self.device
        )
        self.topK_ee_offset = torch.zeros(
            (self.num_combos, cfg.k, 3), device=self.device
        )
        self.topK_ee_angle = torch.zeros(
            (self.num_combos, cfg.k, 3), device=self.device
        )
        self.topK_obj_dist_palm = torch.zeros(
            (self.num_combos, cfg.k, 3), device=self.device
        )
        self.topK_obj_angle_palm = torch.zeros(
            (self.num_combos, cfg.k, 3), device=self.device
        )
        # Individual test scores for each grasp (12 tests per grasp)
        self.topK_test_scores = torch.zeros(
            (self.num_combos, cfg.k, 12), device=self.device
        )

        # Episode counters per combo
        self.episodes_per_combo = torch.zeros(self.num_combos, dtype=torch.int64, device=self.device)

        # Single-shot mode tracking: which envs have completed their one run
        self.env_completed = torch.zeros(self.scene.num_envs, dtype=torch.bool, device=self.device)
        self.env_final_scores = torch.zeros(self.scene.num_envs, dtype=torch.float, device=self.device)
        self.env_completed_object_pose = torch.zeros(self.scene.num_envs, 7, dtype=torch.float, device=self.device)  # Store object pose when completed
        self.env_completed_hand_pose = torch.zeros(self.scene.num_envs, 7, dtype=torch.float, device=self.device)  # Store hand pose (moved away) when completed
        self._initial_reset_done = False  # Track if initial reset has happened

        # Defer initial position update - will be done after first sim step
        # With replicate_physics=False, data isn't available until simulation runs
        self.prev_pos_b = None
        self._initial_update_done = False

        # Finger multiplier for actuation speed randomization (VECTORIZED)
        # Applied to ALL joints (including thumb) - thumb is delayed via wave ordering instead
        self.finger_multiplyer = torch.ones(
            (self.scene.num_envs, max_waves, self.num_joints),
            dtype=torch.float,
            device=self.device
        )

        # Finger multiplier will be set from optimizer via set_grasp_configs() and _reset_idx()
        # No random initialization needed - always uses optimizer values
        
        # === GRASP CONFIG SUPPORT FOR NESTED OPTIMIZATION ===
        # Per-environment grasp parameters (set by set_grasp_configs())
        # Always required in nested optimization mode - no random sampling
        self.grasp_obj_dist_palm = torch.zeros((self.scene.num_envs, 3), device=self.device)  # [num_envs, 3]
        self.grasp_obj_angle_palm = torch.zeros((self.scene.num_envs, 3), device=self.device)  # [num_envs, 3]
        self.grasp_finger_spread = torch.zeros(self.scene.num_envs, device=self.device)  # [num_envs]
        self.grasp_finger_m_rand = torch.ones(self.scene.num_envs, device=self.device)  # [num_envs]

    def _setup_scene(self):
        """Set up the scene with multiple hands and objects."""
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.swapped_object = RigidObject(self.cfg.swapped_object_cfg)
        self.ghost_object = RigidObject(self.cfg.ghost_object_cfg)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(),
            translation=[0, 0.0, -1.0]
        )

        # NOTE: Do NOT call clone_environments when using MultiAssetSpawnerCfg!
        # The spawner already creates prims for all environments, and clone_environments
        # incorrectly merges env_0's content into all other environments.
        # self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["swapped_object"] = self.swapped_object
        self.scene.rigid_objects["ghost_object"] = self.ghost_object

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Register partial physics replicator for faster scene creation
        # Only if we have more envs than unique hand-task combos
        num_combos = self.cfg.num_hands * self.cfg.num_tasks
        if self.scene.num_envs > num_combos:
            self._register_partial_replicator()

    def set_grasp_configs(
        self,
        obj_dist_palm: torch.Tensor,
        obj_angle_palm: torch.Tensor,
        finger_spread: torch.Tensor,
        finger_m_rand: torch.Tensor,
    ):
        """
        Set per-environment grasp configurations for nested optimization.
        
        Always required in nested optimization mode. Call this before reset() to apply new grasp configurations.
        
        Args:
            obj_dist_palm: [num_envs, 3] - object distance from palm (x, y, z)
            obj_angle_palm: [num_envs, 3] - object angle relative to palm (roll, pitch, yaw)
            finger_spread: [num_envs] - finger spreading angle
            finger_m_rand: [num_envs] - finger actuation speed multiplier
        """
        self.grasp_obj_dist_palm = obj_dist_palm.to(self.device)
        self.grasp_obj_angle_palm = obj_angle_palm.to(self.device)
        self.grasp_finger_spread = finger_spread.to(self.device)
        self.grasp_finger_m_rand = finger_m_rand.to(self.device)

    def _register_partial_replicator(self):
        """
        Register a custom PhysX replicator that handles multiple source groups.

        For each unique hand-task combination, we call replicate() with that combo's
        source environment (env_k where k = combo_id) to replicate to all other
        environments with the same combo.

        This dramatically speeds up scene creation by avoiding redundant USD parsing.
        With replicate_physics=False, each environment is parsed independently.
        With partial replication, we only parse once per unique combo and replicate
        the physics representation to all other environments with the same assets.

        Expected speedup: ~100x (e.g., 886s -> 8s for 3072 envs with 24 unique combos)
        """
        import time
        t0 = time.time()

        stage = self.scene.stage
        stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()

        num_combos = self.cfg.num_hands * self.cfg.num_tasks
        num_envs = self.scene.num_envs

        # Build mapping: combo_id -> list of env indices with that combo
        # With MultiUsdFileCfg(random_choice=False), envs are interleaved:
        # env_0 -> combo_0, env_1 -> combo_1, ..., env_23 -> combo_23,
        # env_24 -> combo_0, env_25 -> combo_1, ...
        combo_to_envs = {}
        for env_idx in range(num_envs):
            combo_id = env_idx % num_combos
            if combo_id not in combo_to_envs:
                combo_to_envs[combo_id] = []
            combo_to_envs[combo_id].append(env_idx)

        if self.cfg.debug:
            print(f"[PARTIAL REPLICATION] Registering replicator for {num_combos} combos, {num_envs} envs")
            for combo_id, envs in combo_to_envs.items():
                print(f"  Combo {combo_id}: {len(envs)} envs (source: env_{envs[0]})")

        def replicationAttachFn(stageId):
            # Exclude /World/envs from normal physics parsing
            # This tells PhysX to not automatically create physics for these prims
            print(f"[PARTIAL REPLICATION] replicationAttachFn called - excluding /World/envs")
            return ["/World/envs"]

        def replicationAttachEndFn(stageId):
            # Called when stage is attached - trigger replications for each combo group
            print(f"[PARTIAL REPLICATION] replicationAttachEndFn called - triggering {num_combos} replications")
            for combo_id in range(num_combos):
                env_indices = combo_to_envs[combo_id]
                if len(env_indices) <= 1:
                    continue  # Nothing to replicate

                source_env_idx = env_indices[0]  # First env is the source
                source_path = f"/World/envs/env_{source_env_idx}"
                num_targets = len(env_indices) - 1  # All except source

                success = get_physx_replicator_interface().replicate(
                    stageId, source_path, num_targets, False, False
                )
                if not success:
                    print(f"WARNING: replicate() failed for combo {combo_id} (source: {source_path})")
            print(f"[PARTIAL REPLICATION] replicationAttachEndFn completed - all {num_combos} combos replicated")

        def hierarchyRenameFn(replicatePath, index):
            # Extract source env index from path: "/World/envs/env_X" -> X
            source_env_idx = int(replicatePath.split("_")[-1])
            combo_id = source_env_idx % num_combos

            # Get the target env index for this replication index
            # index is 0-based, so index=0 means the first target (second env in the combo)
            env_indices = combo_to_envs[combo_id]
            target_env_idx = env_indices[index + 1]  # +1 because source is env_indices[0]

            return f"/World/envs/env_{target_env_idx}"

        # Register replicator - the framework will call attach_stage_to_usd_context()
        # after _setup_scene() returns, which will trigger our callbacks
        # NOTE: This requires create_stage_in_memory=True in SimulationCfg
        get_physx_replicator_interface().register_replicator(
            stage_id, replicationAttachFn, replicationAttachEndFn, hierarchyRenameFn
        )

        print(f"[PARTIAL REPLICATION] Replicator registered in {time.time() - t0:.2f}s")

    def _ensure_object_init_pos(self):
        """Lazily initialize object_init_pos after simulation has been stepped."""
        if not self._object_init_pos_initialized:
            self.object_init_pos = torch.nn.functional.pad(
                self.object.data.body_link_state_w[:, 0, :3].clone(), (0, 4)
            )
            self._object_init_pos_initialized = True

    def get_update(self):
        """Update state variables from simulation."""
        self._ensure_object_init_pos()

        # Initialize prev_pos_b on first call
        if not self._initial_update_done:
            self.cur_pos_b = self.object.data.body_link_state_w[:, 0, :3] - self.object_init_pos[:, :3]
            self.cur_quat_b = self.object.data.body_link_state_w[:, 0, 3:7].float()
            self.cur_pos_b = self.cur_pos_b + quat_apply(self.cur_quat_b, self.obj_dist)
            self.prev_pos_b = self.cur_pos_b.clone()
            self._initial_update_done = True

        self.cur_pos_b = self.object.data.body_link_state_w[:, 0, :3] - self.object_init_pos[:, :3]
        self.cur_quat_b = self.object.data.body_link_state_w[:, 0, 3:7].float()
        self.cur_pos_b = self.cur_pos_b + quat_apply(self.cur_quat_b, self.obj_dist)

        self.swp_pos_b = self.swapped_object.data.body_link_state_w[:, 0, :3] - self.object_init_pos[:, :3]
        self.swp_quat_b = self.swapped_object.data.body_link_state_w[:, 0, 3:7].float()
        self.swp_pos_b = self.swp_pos_b + quat_apply(self.swp_quat_b, self.obj_dist)

        self.goal_pos_b = self.ghost_object.data.body_link_state_w[:, 0, :3] - self.object_init_pos[:, :3]
        self.goal_quat_b = self.ghost_object.data.body_link_state_w[:, 0, 3:7].float()
        self.goal_pos_b = self.goal_pos_b + quat_apply(self.goal_quat_b, self.obj_dist)

        self.joint_effort = self.robot.data.applied_torque

    def _get_transforms_for_env(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Get transforms for specific environments based on their task."""
        return self.initial_transforms_per_env[env_ids]

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.get_update()

        # Compute phase masks
        pre_mask = self.delayed > 0
        due_mask = self.delayed == 0
        post_mask = self.delayed < 0

        if pre_mask.any():
            pre_envs = pre_mask.nonzero(as_tuple=False).squeeze(-1)
            # In single-shot mode, filter out already-completed envs
            if self.cfg.single_shot_mode:
                not_completed = ~self.env_completed[pre_envs]
                pre_envs = pre_envs[not_completed]
            if pre_envs.numel() > 0:
                self._pre_sequence(pre_envs)

        if due_mask.any():
            due_envs = due_mask.nonzero(as_tuple=False).squeeze(-1)
            # In single-shot mode, filter out already-completed envs
            if self.cfg.single_shot_mode:
                not_completed = ~self.env_completed[due_envs]
                due_envs = due_envs[not_completed]
            if due_envs.numel() > 0:
                self._due_sequence(due_envs)

        if post_mask.any():
            post_envs = post_mask.nonzero(as_tuple=False).squeeze(-1)
            # In single-shot mode, filter out already-completed envs
            if self.cfg.single_shot_mode:
                not_completed = ~self.env_completed[post_envs]
                post_envs = post_envs[not_completed]
            if post_envs.numel() > 0:
                self._post_sequence(post_envs)

        self.delayed -= 1
        
        # In single-shot mode, continuously fix object positions for completed environments
        # to prevent them from floating/falling due to physics
        if self.cfg.single_shot_mode:
            completed_mask = self.env_completed
            if completed_mask.any():
                completed_envs = completed_mask.nonzero(as_tuple=False).squeeze(-1)
                n_completed = len(completed_envs)
                if n_completed > 0:
                    # Use stored object position from when environment completed
                    completed_pose = self.env_completed_object_pose[completed_envs]
                    
                    # Continuously fix position and zero velocity to prevent physics drift
                    self.swapped_object.write_root_pose_to_sim(completed_pose, env_ids=completed_envs)
                    self.swapped_object.write_root_velocity_to_sim(self._zero_vel_full[:n_completed], env_ids=completed_envs)

    def _pre_sequence(self, pre_envs: torch.Tensor) -> None:
        T = self._get_transforms_for_env(pre_envs) + self.object_init_pos[pre_envs, :7]
        n = pre_envs.shape[0]
        self.object.write_root_pose_to_sim(T, env_ids=pre_envs)
        self.object.write_root_velocity_to_sim(self._zero_vel_full[:n], env_ids=pre_envs)
        self.swapped_object.write_root_pose_to_sim(T + self.swap_add[pre_envs, :7], env_ids=pre_envs)
        self.swapped_object.write_root_velocity_to_sim(self._zero_vel_full[:n], env_ids=pre_envs)
        self.update_finger_groups()

    def _due_sequence(self, due_envs: torch.Tensor) -> None:
        # Copy snapshot values
        self.opt_object_pos[due_envs] = self.cur_pos_b[due_envs].clone()
        self.opt_object_quat[due_envs] = self.cur_quat_b[due_envs].float()
        self.opt_robot_joint_pos[due_envs] = self.robot.data.joint_pos[due_envs].clone()
        self.opt_robot_target_pos[due_envs] = self.target_hand_pos[due_envs].clone()
        self.opt_ee_offset[due_envs] = self.ee_pos_offset[due_envs].clone()
        self.opt_ee_angle[due_envs] = self.ee_angle_offset[due_envs].clone()
        self.opt_obj_dist_palm[due_envs] = self.obj_dist_palm[due_envs].clone()
        self.opt_obj_angle_palm[due_envs] = self.obj_angle_palm[due_envs].clone()


        T = self._get_transforms_for_env(due_envs) + self.object_init_pos[due_envs, :7]
        n = due_envs.shape[0]
        self.object.write_root_pose_to_sim(T + self.swap_add[due_envs, :7], env_ids=due_envs)
        self.object.write_root_velocity_to_sim(self._zero_vel_full[:n], env_ids=due_envs)
        self.swapped_object.write_root_pose_to_sim(T, env_ids=due_envs)
        self.swapped_object.write_root_velocity_to_sim(self._zero_vel_full[:n], env_ids=due_envs)

        self.test_start[due_envs] = self.timestep[due_envs].clone()

    def _post_sequence(self, post_envs: torch.Tensor) -> None:
        delay_frames = int(self.cfg.test_delay / self.cfg.dt)
        raw_score = (self.timestep[post_envs] - self.test_start[post_envs]) - delay_frames
        self.test_score[post_envs, self.test_cur[post_envs]] = torch.clamp(raw_score, min=0)
        apply_force = self.tests[self.test_cur[post_envs]] * \
                     (self.test_score[post_envs, self.test_cur[post_envs]].unsqueeze(1) /
                      (self.cfg.time_per_test / self.cfg.dt))

        obj_inv = quat_inv(self.swp_quat_b[post_envs])
        apply_force[:, :3] = quat_apply(obj_inv, apply_force[:, :3])
        apply_force = apply_force.unsqueeze(1)
        
        # In single-shot mode, don't apply forces to completed environments
        if self.cfg.single_shot_mode:
            not_completed_mask = ~self.env_completed[post_envs]
            active_post_envs = post_envs[not_completed_mask]
            if active_post_envs.numel() > 0:
                active_apply_force = apply_force[not_completed_mask]
                self.swapped_object.permanent_wrench_composer.set_forces_and_torques(
                    forces=active_apply_force[:, :, :3],
                    torques=active_apply_force[:, :, 3:6],
                    env_ids=active_post_envs
                )
        else:
            self.swapped_object.permanent_wrench_composer.set_forces_and_torques(
                forces=apply_force[:, :, :3],
                torques=apply_force[:, :, 3:6],
                env_ids=post_envs
            )
        reset_mask = self.test_score[post_envs, self.test_cur[post_envs]] == 0

        if reset_mask.any():
            reset_envs = post_envs[reset_mask]
            T = self._get_transforms_for_env(reset_envs) + self.object_init_pos[reset_envs, :7]
            n = reset_envs.shape[0]
            self.robot.write_joint_state_to_sim(
                self.opt_robot_joint_pos[reset_envs],
                self._zero_joint_vel[:n],
                env_ids=reset_envs
            )
            self.robot.set_joint_position_target(
                self.opt_robot_target_pos[reset_envs],
                env_ids=reset_envs
            )
            self.swapped_object.write_root_pose_to_sim(T, env_ids=reset_envs)
            self.swapped_object.write_root_velocity_to_sim(self._zero_vel_full[:n], env_ids=reset_envs)
            self.object.write_root_pose_to_sim(T + self.swap_add[reset_envs, :7], env_ids=reset_envs)
            self.object.write_root_velocity_to_sim(self._zero_vel_full[:n], env_ids=reset_envs)

        # Evaluate drift
        pos_diff = torch.linalg.norm(self.swp_pos_b[post_envs] - self.opt_object_pos[post_envs], dim=-1)
        dist_mask = pos_diff >= self.cfg.max_test_pos
        opt_quat = self.opt_object_quat[post_envs]
        swp_quat = self.swp_quat_b[post_envs]
        relative_quat = quat_mul(quat_inv(opt_quat), swp_quat)
        # Use abs(w) to account for q and -q representing the same rotation (gives the minimal angle in [0, pi])
        w = torch.abs(relative_quat[:, 0])
        angle_diff = 2 * torch.acos(torch.clamp(w, min=0.0, max=1.0))
        # angle_diff = 2 * torch.acos(torch.clamp(relative_quat[:, 0], min=-1.0, max=1.0))
        angle_mask = angle_diff > self.cfg.max_test_angle
        time_mask = self.test_score[post_envs, self.test_cur[post_envs]] > self.cfg.time_per_test / self.cfg.dt


        next_test_combined = dist_mask | angle_mask | time_mask
        if next_test_combined.any():
            next_test_envs = post_envs[next_test_combined]
            # Cap test_score to max frames when test ends (fixes score > 1.0 bug)
            max_frames = int(self.cfg.time_per_test / self.cfg.dt)
            current_test_idx = self.test_cur[next_test_envs]
            self.test_score[next_test_envs, current_test_idx] = torch.clamp(
                self.test_score[next_test_envs, current_test_idx],
                max=max_frames
            )
            self.test_start[next_test_envs] = self.timestep[next_test_envs]
            self.test_cur[next_test_envs] += 1

    def update_finger_groups(self):
        """Update finger group closing state."""
        self.finger_wait -= 1
        update_pos = self.finger_wait >= 0
        self.prev_pos_b[update_pos] = self.cur_pos_b[update_pos]

        max_waves = self.finger_groups.shape[1]
        active = (self.current_finger_groups < max_waves) & (self.finger_wait < 0)
        if not active.any():
            return

        env_indices = active.nonzero(as_tuple=False).squeeze(-1)
        finger_group_delay_frames = int(self.cfg.finger_group_delay / self.cfg.sim.dt)

        neg_mask = self.current_finger_groups[env_indices] == -1
        neg_processed = torch.zeros(env_indices.shape[0], dtype=torch.bool, device=self.device)

        if neg_mask.any():
            env_neg = env_indices[neg_mask]
            self.ee_pos_offset[env_neg, 0] -= self.cfg.palm_speed
            pos_diff = self.cur_pos_b[env_neg] - self.prev_pos_b[env_neg]
            recoil_mask = (pos_diff * pos_diff).sum(dim=-1) >= self.cfg.object_threshold ** 2

            if recoil_mask.any():
                env_recoil = env_neg[recoil_mask]
                self.ee_pos_offset[env_recoil, 0] += self.cfg.palm_speed * self.cfg.palm_recoil
                self.finger_wait[env_recoil] = finger_group_delay_frames
                self.current_finger_groups[env_recoil] += 1
                neg_processed[neg_mask] = recoil_mask

        nonneg_mask = (self.current_finger_groups[env_indices] >= 0) & (~neg_processed)
        if nonneg_mask.any():
            env_nonneg = env_indices[nonneg_mask]
            current_group_idx = self.current_finger_groups[env_nonneg]
            joint_indices = self.finger_groups[env_nonneg, current_group_idx, :]

            update_mask = (torch.abs(self.joint_effort[env_nonneg]) * joint_indices) > self.torque_stop[env_nonneg]
            self.finger_groups[env_nonneg, current_group_idx, :] *= ~update_mask

            torque_ratio = torch.abs(self.joint_effort[env_nonneg]) / self.torque_stop[env_nonneg]
            speed = torch.where(
                torque_ratio < self.cfg.torque_approach_threshold,
                self.cfg.joint_speed_fast,
                self.cfg.joint_speed_slow
            )

            self.target_hand_pos[env_nonneg, :] += (
                speed *
                self.closing_directions[env_nonneg] *
                self.finger_groups[env_nonneg, current_group_idx, :] *
                self.finger_multiplyer[env_nonneg, current_group_idx, :]
            )

            all_done = (joint_indices == 0).all(dim=1)
            if all_done.any():
                self.current_finger_groups[env_nonneg[all_done]] += 1

        min_buffer_frames = int(self.cfg.min_post_close_buffer / (self.cfg.sim.dt * self.cfg.decimation))
        done_mask = (self.current_finger_groups[env_indices] >= max_waves) & \
                   (self.delayed[env_indices] > min_buffer_frames)
        if done_mask.any():
            self.delayed[env_indices[done_mask]] = min_buffer_frames

    def _apply_action(self) -> None:
        T = self._get_transforms_for_env(torch.arange(self.scene.num_envs, device=self.device)) + \
            self.object_init_pos[:, :7]
        self.ghost_object.write_root_pose_to_sim(T)

        object_pos_b, object_quat_b = T[:, :3], T[:, 3:7]
        pre_rand_pos = object_pos_b + quat_apply(object_quat_b, self.ee_pos_offset)
        rand_pos = quat_apply(object_quat_b, self.obj_dist_palm)
        self.target_ee_pos_b = pre_rand_pos + rand_pos

        quat_change = quat_from_euler_xyz(
            self.ee_angle_offset[:, 0],
            self.ee_angle_offset[:, 1],
            self.ee_angle_offset[:, 2]
        )
        pre_rand_quat = quat_mul(object_quat_b, quat_change)
        rand_quat_change = quat_from_euler_xyz(
            self.obj_angle_palm[:, 0],
            self.obj_angle_palm[:, 1],
            self.obj_angle_palm[:, 2]
        )
        computed_quat = quat_mul(pre_rand_quat, rand_quat_change)
        self.target_ee_quat_b = quat_mul(self.base_rotation_per_env, computed_quat)

        target_root_pose = torch.cat([self.target_ee_pos_b, self.target_ee_quat_b], dim=-1)

        # In single-shot mode, don't keep driving completed envs (prevents post-finish jitter/contact).
        if self.cfg.single_shot_mode:
            active_envs = (~self.env_completed).nonzero(as_tuple=False).squeeze(-1)
            if active_envs.numel() > 0:
                self.robot.write_root_pose_to_sim(target_root_pose[active_envs], env_ids=active_envs)
                self.robot.set_joint_position_target(self.target_hand_pos[active_envs], env_ids=active_envs)

            completed_envs = self.env_completed.nonzero(as_tuple=False).squeeze(-1)
            if completed_envs.numel() > 0:
                # Keep completed hands at their stored “moved away” pose
                self.robot.write_root_pose_to_sim(self.env_completed_hand_pose[completed_envs], env_ids=completed_envs)
                # No need to keep updating joint targets for completed envs
        else:
            self.robot.write_root_pose_to_sim(target_root_pose)
        self.robot.set_joint_position_target(self.target_hand_pos)
        self.timestep[self.delayed < 0] += 1

    def _get_observations(self) -> dict:
        return {}

    def _get_rewards(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        test_done = self.test_cur >= self.tests.shape[0]
        # In single-shot mode, also consider already-completed envs as done
        if self.cfg.single_shot_mode:
            test_done = test_done | self.env_completed
        return test_done, torch.zeros_like(test_done)

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        if len(env_ids) == 0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device)

        # In single-shot mode, filter out already-completed envs
        if self.cfg.single_shot_mode:
            not_completed = ~self.env_completed[env_ids]
            env_ids = env_ids[not_completed]
            if len(env_ids) == 0:
                return

        super()._reset_idx(env_ids)

        # Ensure object_init_pos is initialized (lazy init after simulation starts)
        self._ensure_object_init_pos()

        # Compute final scores for completed episodes
        new_scores = self.test_score[env_ids].sum(dim=-1).float() / \
                    (self.cfg.time_per_test / self.cfg.dt * 12)

        # Update topK per combo (vectorized)
        reset_combos = self.combo_indices[env_ids]
        max_new = self.cfg.envs_per_combo
        combined_k = self.cfg.k + max_new

        combined_scores = torch.full(
            (self.num_combos, combined_k), -float("inf"), device=self.device
        )
        combined_joint_pos = torch.zeros(
            (self.num_combos, combined_k, self.num_joints), device=self.device
        )
        combined_target_pos = torch.zeros(
            (self.num_combos, combined_k, self.num_joints), device=self.device
        )
        combined_ee_pos = torch.zeros(
            (self.num_combos, combined_k, 3), device=self.device
        )
        combined_ee_quat = torch.zeros(
            (self.num_combos, combined_k, 4), device=self.device
        )
        combined_ee_offset = torch.zeros(
            (self.num_combos, combined_k, 3), device=self.device
        )
        combined_ee_angle = torch.zeros(
            (self.num_combos, combined_k, 3), device=self.device
        )
        combined_obj_dist = torch.zeros(
            (self.num_combos, combined_k, 3), device=self.device
        )
        combined_obj_angle = torch.zeros(
            (self.num_combos, combined_k, 3), device=self.device
        )
        combined_test_scores = torch.zeros(
            (self.num_combos, combined_k, 12), device=self.device
        )

        combined_scores[:, :self.cfg.k] = self.topK_scores
        combined_joint_pos[:, :self.cfg.k] = self.topK_joint_pos
        combined_target_pos[:, :self.cfg.k] = self.topK_target_pos
        combined_ee_pos[:, :self.cfg.k] = self.topK_ee_pos
        combined_ee_quat[:, :self.cfg.k] = self.topK_ee_quat
        combined_ee_offset[:, :self.cfg.k] = self.topK_ee_offset
        combined_ee_angle[:, :self.cfg.k] = self.topK_ee_angle
        combined_obj_dist[:, :self.cfg.k] = self.topK_obj_dist_palm
        combined_obj_angle[:, :self.cfg.k] = self.topK_obj_angle_palm
        combined_test_scores[:, :self.cfg.k] = self.topK_test_scores

        env_pos_in_combo = (env_ids // self.num_combos).to(dtype=torch.int64)
        scatter_idx = self.cfg.k + env_pos_in_combo

        combined_scores[reset_combos, scatter_idx] = new_scores
        combined_joint_pos[reset_combos, scatter_idx] = self.opt_robot_joint_pos[env_ids]
        combined_target_pos[reset_combos, scatter_idx] = self.opt_robot_target_pos[env_ids]
        combined_ee_pos[reset_combos, scatter_idx] = self.target_ee_pos_b[env_ids]
        combined_ee_quat[reset_combos, scatter_idx] = self.target_ee_quat_b[env_ids]
        combined_ee_offset[reset_combos, scatter_idx] = self.opt_ee_offset[env_ids]
        combined_ee_angle[reset_combos, scatter_idx] = self.opt_ee_angle[env_ids]
        combined_obj_dist[reset_combos, scatter_idx] = self.opt_obj_dist_palm[env_ids]
        combined_obj_angle[reset_combos, scatter_idx] = self.opt_obj_angle_palm[env_ids]
        combined_test_scores[reset_combos, scatter_idx] = self.test_score[env_ids].float()

        topk_vals, topk_idx = torch.topk(combined_scores, k=self.cfg.k, dim=1)
        self.topK_scores = topk_vals

        topk_idx_joints = topk_idx.unsqueeze(-1).expand(-1, -1, self.num_joints)
        topk_idx_vec3 = topk_idx.unsqueeze(-1).expand(-1, -1, 3)
        topk_idx_vec4 = topk_idx.unsqueeze(-1).expand(-1, -1, 4)

        self.topK_joint_pos = torch.gather(combined_joint_pos, 1, topk_idx_joints)
        self.topK_target_pos = torch.gather(combined_target_pos, 1, topk_idx_joints)
        self.topK_ee_pos = torch.gather(combined_ee_pos, 1, topk_idx_vec3)
        self.topK_ee_quat = torch.gather(combined_ee_quat, 1, topk_idx_vec4)
        self.topK_ee_offset = torch.gather(combined_ee_offset, 1, topk_idx_vec3)
        self.topK_ee_angle = torch.gather(combined_ee_angle, 1, topk_idx_vec3)
        self.topK_obj_dist_palm = torch.gather(combined_obj_dist, 1, topk_idx_vec3)
        self.topK_obj_angle_palm = torch.gather(combined_obj_angle, 1, topk_idx_vec3)
        topk_idx_12 = topk_idx.unsqueeze(-1).expand(-1, -1, 12)
        self.topK_test_scores = torch.gather(combined_test_scores, 1, topk_idx_12)

        # Update episode count per combo (vectorized)
        self.episodes_per_combo.scatter_add_(
            0,
            reset_combos,
            torch.ones_like(reset_combos, dtype=self.episodes_per_combo.dtype)
        )

        # In single-shot mode: store final scores, mark envs as completed, and skip actual reset
        # BUT only after the initial reset (first reset is for setup, not completion)
        if self.cfg.single_shot_mode and self._initial_reset_done:
            self.env_final_scores[env_ids] = new_scores
            self.env_completed[env_ids] = True
            
            # Store object position when environment completes to fix it later
            n_completed = len(env_ids)
            completed_pos = self.swp_pos_b[env_ids] + self.object_init_pos[env_ids, :3]
            completed_quat = self.swp_quat_b[env_ids]
            completed_pose = torch.cat([completed_pos, completed_quat], dim=-1)
            self.env_completed_object_pose[env_ids] = completed_pose
            
            # Set object to current position with zero velocity to prevent physics drift
            self.swapped_object.write_root_pose_to_sim(completed_pose, env_ids=env_ids)
            self.swapped_object.write_root_velocity_to_sim(self._zero_vel_full[:n_completed], env_ids=env_ids)

            # Also pin the non-swapped object (some visualizations reference it)
            self.object.write_root_pose_to_sim(completed_pose + self.swap_add[env_ids, :7], env_ids=env_ids)
            self.object.write_root_velocity_to_sim(self._zero_vel_full[:n_completed], env_ids=env_ids)

            # Move the hand away immediately to break contact and stop jitter.
            # Offset: -0.15m in Z in world frame.
            hand_pose = torch.cat([self.target_ee_pos_b[env_ids], self.target_ee_quat_b[env_ids]], dim=-1).clone()
            hand_pose[:, 2] -= 0.15
            self.env_completed_hand_pose[env_ids] = hand_pose
            self.robot.write_root_pose_to_sim(hand_pose, env_ids=env_ids)
            
            return  # Don't reset the environment state

        # Reset state for these envs
        self.timestep[env_ids] = 0
        self.delayed[env_ids] = int(self.cfg.delay / (self.cfg.sim.dt * self.cfg.decimation))

        T = self._get_transforms_for_env(env_ids) + self.object_init_pos[env_ids, :7]
        n = len(env_ids)
        self.ghost_object.write_root_pose_to_sim(T, env_ids=env_ids)
        self.object.write_root_pose_to_sim(T, env_ids=env_ids)
        self.object.write_root_velocity_to_sim(self._zero_vel_full[:n], env_ids=env_ids)
        self.swapped_object.write_root_pose_to_sim(T + self.swap_add[env_ids, :7], env_ids=env_ids)
        self.swapped_object.write_root_velocity_to_sim(self._zero_vel_full[:n], env_ids=env_ids)

        # Reset finger state
        self.finger_wait[env_ids] = int(self.cfg.finger_group_delay / self.cfg.sim.dt)
        self.current_finger_groups[env_ids] = -1

        # Reset joint positions to default
        self.target_hand_pos[env_ids] = self.robot.data.default_joint_pos[env_ids, :].clone()

        # === VECTORIZED FANNING/AXIAL JOINT RANDOMIZATION (no loops over envs) ===
        hand_indices = self.env_to_hand[env_ids]  # [n]

        # Get per-env masks and data via indexing
        fanning_mask = self.fanning_mask_per_hand[hand_indices]  # [n, max_joints]
        fan_factors = self.fanning_fan_factors_per_hand[hand_indices]  # [n, max_joints]
        thumb_mask = self.thumb_fanning_mask_per_hand[hand_indices]  # [n, max_joints]
        thumb_base = self.thumb_base_angles_per_hand[hand_indices]  # [n, max_joints]
        axial_mask = self.axial_mask_per_hand[hand_indices]  # [n, max_joints]
        axial_signs = self.axial_alt_signs_per_hand[hand_indices]  # [n, max_joints]
        lower_limits = self.joint_lower_limits_per_hand[hand_indices]  # [n, max_joints]
        upper_limits = self.joint_upper_limits_per_hand[hand_indices]  # [n, max_joints]
        # From robot_builder: thumb first closing at -1.2, thumb axial at midpoint
        thumb_first_closing_mask = self.thumb_first_closing_mask_per_hand[hand_indices]  # [n, max_joints]
        thumb_first_closing_pos = self.thumb_first_closing_pos_per_hand[hand_indices]  # [n, max_joints]
        thumb_axial_mask = self.thumb_axial_mask_per_hand[hand_indices]  # [n, max_joints]
        thumb_axial_midpoints = self.thumb_axial_midpoints_per_hand[hand_indices]  # [n, max_joints]

        # --- Non-thumb fanning joints ---
        # Use provided finger_spread from optimizer
        spread_mag = self.grasp_finger_spread[env_ids]  # [n]
        
        use_fanning = torch.rand(n, device=self.device) > 0.5  # [n]
        direction = (torch.rand(n, device=self.device) > 0.5).float() * 2 - 1  # [n] -> +/-1

        # Fanning angles: either fan pattern or uniform
        fanning_angles = torch.where(
            use_fanning.unsqueeze(1),  # [n, 1]
            spread_mag.unsqueeze(1) * fan_factors * direction.unsqueeze(1),  # fan pattern
            spread_mag.unsqueeze(1) * direction.unsqueeze(1)  # uniform
        )  # [n, max_joints]
        fanning_angles = torch.clamp(fanning_angles, lower_limits, upper_limits)

        # --- Thumb fanning joints ---
        thumb_rand = (torch.rand(n, 1, device=self.device) - 0.5) * self.grasp_finger_spread[env_ids].unsqueeze(1) * 2
        thumb_angles = thumb_base + thumb_rand  # [n, max_joints] (broadcasts)
        thumb_angles = torch.clamp(thumb_angles, lower_limits, upper_limits)

        # --- Axial joints ---
        use_alternating = torch.rand(n, device=self.device) > 0.5  # [n]
        alt_direction = (torch.rand(n, device=self.device) > 0.5).float() * 2 - 1  # [n]
        axial_spread = (torch.rand(n, device=self.device) - 0.5) * self.grasp_finger_spread[env_ids] * 0.5
        axial_magnitude = torch.rand(n, device=self.device) * self.grasp_finger_spread[env_ids] * 0.5

        # Axial angles: either alternating pattern or uniform spread
        axial_angles = torch.where(
            use_alternating.unsqueeze(1),  # [n, 1]
            axial_magnitude.unsqueeze(1) * alt_direction.unsqueeze(1) * axial_signs,  # alternating
            axial_spread.unsqueeze(1).expand(-1, self.num_joints)  # uniform
        )  # [n, max_joints]
        axial_angles = torch.clamp(axial_angles, lower_limits, upper_limits)

        # Apply all joint randomizations (vectorized where operations)
        # Order matters: more specific masks applied last can override earlier ones
        target_pos = self.target_hand_pos[env_ids]
        target_pos = torch.where(fanning_mask, fanning_angles, target_pos)
        target_pos = torch.where(thumb_mask, thumb_angles, target_pos)
        target_pos = torch.where(axial_mask, axial_angles, target_pos)
        # From robot_builder: thumb first closing starts at -1.2 (pre-curled)
        target_pos = torch.where(thumb_first_closing_mask, thumb_first_closing_pos, target_pos)
        # From robot_builder: thumb axial joints start at midpoint of limits
        target_pos = torch.where(thumb_axial_mask, thumb_axial_midpoints, target_pos)
        self.target_hand_pos[env_ids] = target_pos

        # Write joint state to sim
        # Ensure tensors are regular (not inference) tensors for Isaac Lab's in-place updates
        # Use detach() to break any inference mode connection
        target_pos = self.target_hand_pos[env_ids].detach().clone()
        target_vel = torch.zeros_like(target_pos).detach()
        self.robot.write_joint_state_to_sim(
            target_pos,
            target_vel,
            None,
            env_ids
        )

        # Reset finger groups (VECTORIZED - no loops)
        self.finger_groups[env_ids] = self.finger_groups_per_hand[hand_indices].clone()

        # Reset finger multiplier with provided value from optimizer for ALL joints
        # (thumb delay is handled via wave ordering, not speed multiplier)
        self.finger_multiplyer[env_ids] = torch.ones(
            (n, self.max_waves, self.num_joints),
            dtype=torch.float,
            device=self.device
        )
        
        # Use provided finger_m_rand from optimizer
        finger_m_rand = self.grasp_finger_m_rand[env_ids].view(n, 1, 1)

        # Apply multiplier to all joints
        self.finger_multiplyer[env_ids] = self.finger_multiplyer[env_ids] * finger_m_rand

        # Reset test state
        self.test_score[env_ids] = 0
        self.test_cur[env_ids] = 0
        self.test_start[env_ids] = 0

        # Reset torque stop
        random_torques = torch.rand(n, device=self.device) * \
                        (self.cfg.torque_max - self.cfg.torque_min) + self.cfg.torque_min
        self.torque_stop[env_ids] = random_torques.unsqueeze(1).repeat(1, self.num_joints)

        # Reset EE offsets
        self.ee_pos_offset[env_ids] = self.ee_offset_pos_base[env_ids].clone()
        self.ee_angle_offset[env_ids] = self.ee_offset_angle_base[env_ids].clone()

        # Set palm offset from grasp configs (always from optimizer in nested mode)
        self.obj_dist_palm[env_ids] = self.grasp_obj_dist_palm[env_ids]
        self.obj_angle_palm[env_ids] = self.grasp_obj_angle_palm[env_ids]

        self.get_update()
        self.prev_pos_b[env_ids] = self.cur_pos_b[env_ids].clone()

        # Mark initial reset as done (for single-shot mode)
        if not self._initial_reset_done:
            self._initial_reset_done = True

    def get_topK_data(self) -> Dict:
        """Get all topK data for saving."""
        # Compute normalized test scores (each test score / max_frames)
        max_frames = self.cfg.time_per_test / self.cfg.dt
        topK_test_scores_normalized = self.topK_test_scores / max_frames

        return {
            'topK_scores': self.topK_scores.cpu(),
            'topK_joint_pos': self.topK_joint_pos.cpu(),
            'topK_target_pos': self.topK_target_pos.cpu(),
            'topK_ee_pos': self.topK_ee_pos.cpu(),
            'topK_ee_quat': self.topK_ee_quat.cpu(),
            'topK_ee_offset': self.topK_ee_offset.cpu(),
            'topK_ee_angle': self.topK_ee_angle.cpu(),
            'topK_obj_dist_palm': self.topK_obj_dist_palm.cpu(),
            'topK_obj_angle_palm': self.topK_obj_angle_palm.cpu(),
            'topK_test_scores_raw': self.topK_test_scores.cpu(),  # Raw frames held per test
            'topK_test_scores_normalized': topK_test_scores_normalized.cpu(),  # 0-1 per test
            'episodes_per_combo': self.episodes_per_combo.cpu(),
            'test_config': {
                'time_per_test': self.cfg.time_per_test,
                'test_delay': self.cfg.test_delay,
                'newton_per_test': self.cfg.newton_per_test,
                'torque_per_test': self.cfg.torque_per_test,
                'max_test_pos': self.cfg.max_test_pos,
                'max_test_angle': self.cfg.max_test_angle,
                'dt': self.cfg.dt,
                'max_frames_per_test': max_frames,
            },
        }

    def get_combo_info(self, combo_idx: int) -> Tuple[str, str]:
        """Get hand name and task name for a combo index."""
        hand_idx = combo_idx // self.cfg.num_tasks
        task_idx = combo_idx % self.cfg.num_tasks
        hand_name = self.cfg.padded_hands[hand_idx].hand_name
        task_name = self.cfg.tasks[task_idx]
        return hand_name, task_name

    def all_envs_completed(self) -> bool:
        """Check if all environments have completed (for single-shot mode).
        
        Returns True if single_shot_mode is enabled and all envs have finished.
        Always returns False if single_shot_mode is disabled.
        """
        if not self.cfg.single_shot_mode:
            return False
        return bool(self.env_completed.all().item())

    def get_completion_status(self) -> Dict:
        """Get completion status for single-shot mode.
        
        Returns dict with:
        - completed_count: number of completed envs
        - total_count: total number of envs
        - completed_ratio: ratio of completed envs
        - all_completed: whether all envs are done
        - env_final_scores: tensor of final scores (only valid for completed envs)
        """
        completed_count = self.env_completed.sum().item()
        total_count = self.scene.num_envs
        return {
            'completed_count': completed_count,
            'total_count': total_count,
            'completed_ratio': completed_count / total_count,
            'all_completed': completed_count == total_count,
            'env_final_scores': self.env_final_scores.cpu(),
            'env_completed': self.env_completed.cpu(),
        }
