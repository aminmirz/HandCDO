"""
URDF Padding utility for multi-hand grasp selection.

Pads URDFs with dummy fixed joints to ensure all hands have the same joint count,
enabling true parallelism with MultiUsdFileCfg in Isaac Lab.
"""

from __future__ import annotations
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from robotouch.hand_config import parse_urdf, HandConfig


@dataclass
class PaddedHandInfo:
    """Information about a padded hand configuration."""
    original_urdf_path: str
    padded_urdf_path: str
    usd_path: str = ""  # Path to converted USD file (set after conversion)
    hand_name: str = ""
    real_joint_mask: List[bool] = field(default_factory=list)  # True for real joints, False for dummy
    real_joint_names: List[str] = field(default_factory=list)  # Original joint names
    num_real_joints: int = 0
    num_total_joints: int = 0
    hand_config: HandConfig = None


def analyze_hands(urdf_folder: str) -> Tuple[int, List[str], Dict[str, HandConfig], Dict[str, Tuple[str, str]]]:
    """
    Scan all URDFs in folder and find the maximum joint count, joint template,
    and canonical kinematic tree (parent-child link mapping for each joint).

    Joint template is the union of all joint names across all hands, sorted alphabetically.
    This ensures consistent ordering for tensor operations.

    The canonical tree maps each joint name to its (parent_link, child_link) pair,
    collected from whichever hand actually has that joint. This ensures all padded
    URDFs share the same kinematic tree topology.

    Args:
        urdf_folder: Path to folder containing hand URDF subfolders

    Returns:
        Tuple of (max_joints, joint_template, hand_configs, canonical_tree)
        - max_joints: Maximum number of joints across all hands
        - joint_template: Sorted list of all unique joint names across all hands
        - hand_configs: Dict mapping hand_name -> HandConfig
        - canonical_tree: Dict mapping joint_name -> (parent_link_name, child_link_name)
    """
    urdf_folder = Path(urdf_folder)

    if not urdf_folder.exists():
        raise ValueError(f"URDF folder does not exist: {urdf_folder}")

    all_joint_names: Set[str] = set()
    hand_configs: Dict[str, HandConfig] = {}
    canonical_tree: Dict[str, Tuple[str, str]] = {}
    max_joints = 0

    # Find all hand subdirectories with hand_robot.urdf
    for hand_dir in sorted(urdf_folder.iterdir()):
        if not hand_dir.is_dir():
            continue

        urdf_path = hand_dir / "urdf_hand_export" / "hand_robot.urdf"
        # urdf_path = hand_dir / "hand_robot.urdf"
        if not urdf_path.exists():
            continue

        hand_name = hand_dir.name
        try:
            hand_config = parse_urdf(str(urdf_path))
            hand_configs[hand_name] = hand_config

            all_joint_names.update(hand_config.joint_names)
            max_joints = max(max_joints, hand_config.num_joints)

            # Extract parent-child link mapping from URDF for canonical tree
            tree = ET.parse(str(urdf_path))
            root = tree.getroot()
            for joint in root.findall('joint'):
                if joint.get('type') == 'revolute':
                    jname = joint.get('name')
                    parent_elem = joint.find('parent')
                    child_elem = joint.find('child')
                    if jname and parent_elem is not None and child_elem is not None:
                        parent_link = parent_elem.get('link')
                        child_link = child_elem.get('link')
                        if jname not in canonical_tree:
                            canonical_tree[jname] = (parent_link, child_link)

        except Exception as e:
            print(f"Warning: Failed to parse {urdf_path}: {e}")
            continue

    if not hand_configs:
        raise ValueError(f"No valid URDFs found in {urdf_folder}")

    # Sort joint names for consistent ordering
    joint_template = sorted(all_joint_names)

    return max_joints, joint_template, hand_configs, canonical_tree


def pad_urdf(
    urdf_path: str,
    joint_template: List[str],
    output_dir: str,
    hand_name: str,
    canonical_tree: Dict[str, Tuple[str, str]] = None,
) -> Tuple[str, List[bool]]:
    """
    Pad URDF with dummy joints to match the joint template.

    Dummy joints mirror the canonical kinematic tree topology so that all
    padded URDFs have identical tree structures. This is required by PhysX's
    ArticulationView which needs matching kinematic trees for batched
    articulations.

    Args:
        urdf_path: Path to original URDF file
        joint_template: List of all joint names that should exist (sorted)
        output_dir: Directory to save padded URDF
        hand_name: Name of the hand (for output filename)
        canonical_tree: Dict mapping joint_name -> (parent_link, child_link).
            Determines the correct parent-child chain for dummy joints.
            If None, falls back to attaching dummy links to base_link.

    Returns:
        Tuple of (padded_urdf_path, real_joint_mask)
        - padded_urdf_path: Path to the padded URDF
        - real_joint_mask: List[bool] where True means joint is real, False means dummy
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Get the original URDF directory for resolving relative paths
    urdf_dir = Path(urdf_path).parent.resolve()

    # Convert all file paths to absolute paths
    # This is necessary because the padded URDF will be in a different directory

    # Mesh files
    for mesh_elem in root.iter('mesh'):
        filename = mesh_elem.get('filename')
        if filename and not Path(filename).is_absolute():
            abs_path = (urdf_dir / filename).resolve()
            mesh_elem.set('filename', str(abs_path))

    # Texture files
    for texture_elem in root.iter('texture'):
        filename = texture_elem.get('filename')
        if filename and not Path(filename).is_absolute():
            abs_path = (urdf_dir / filename).resolve()
            texture_elem.set('filename', str(abs_path))

    # Get existing revolute joint names
    existing_joints: Set[str] = set()
    for joint in root.findall('joint'):
        if joint.get('type') == 'revolute':
            existing_joints.add(joint.get('name'))

    # Build real joint mask: True if joint exists in original URDF
    real_mask = [name in existing_joints for name in joint_template]

    # Find base link name (for attaching dummy joints when canonical tree is unavailable)
    base_link_name = "base_link"
    for link in root.findall('link'):
        link_name = link.get('name')
        if link_name and ('base' in link_name.lower() or 'palm' in link_name.lower()):
            base_link_name = link_name
            break

    def _ensure_link_exists(link_name: str):
        """Create a dummy link with minimal geometry if it doesn't exist."""
        if root.find(f".//link[@name='{link_name}']") is None:
            dummy_link = ET.SubElement(root, 'link', name=link_name)
            # Add minimal inertial to avoid physics issues
            inertial = ET.SubElement(dummy_link, 'inertial')
            ET.SubElement(inertial, 'mass', value="0.0001")
            ET.SubElement(inertial, 'inertia',
                          ixx="0.0000001", ixy="0", ixz="0",
                          iyy="0.0000001", iyz="0", izz="0.0000001")
            # Add minimal visual/collision geometry to prevent unresolved
            # USD prim references when the converter creates sublayers
            visual = ET.SubElement(dummy_link, 'visual')
            vis_geom = ET.SubElement(visual, 'geometry')
            ET.SubElement(vis_geom, 'sphere', radius="0.0001")
            collision = ET.SubElement(dummy_link, 'collision')
            col_geom = ET.SubElement(collision, 'geometry')
            ET.SubElement(col_geom, 'sphere', radius="0.0001")

    # Add dummy joints for missing joints, preserving canonical tree topology
    for idx, joint_name in enumerate(joint_template):
        if joint_name in existing_joints:
            continue

        # Determine parent and child links from canonical tree
        if canonical_tree and joint_name in canonical_tree:
            parent_link_name_for_joint, child_link_name = canonical_tree[joint_name]
        else:
            # Fallback: attach to base link with index-based dummy name
            parent_link_name_for_joint = base_link_name
            child_link_name = f"dummy_link_{idx:02d}"

        # Ensure the child link exists (create dummy if needed)
        _ensure_link_exists(child_link_name)

        # Create dummy revolute joint with zero range (effectively fixed)
        dummy_joint = ET.SubElement(root, 'joint', name=joint_name, type='revolute')
        ET.SubElement(dummy_joint, 'parent', link=parent_link_name_for_joint)
        ET.SubElement(dummy_joint, 'child', link=child_link_name)
        ET.SubElement(dummy_joint, 'origin', xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(dummy_joint, 'axis', xyz="0 0 1")
        # Zero range means joint is effectively locked
        ET.SubElement(dummy_joint, 'limit', lower="0", upper="0", effort="0", velocity="0")

    # Save padded URDF
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{hand_name}_padded.urdf"
    tree.write(str(output_path), xml_declaration=True, encoding='utf-8')

    return str(output_path), real_mask


def prepare_padded_hands(
    urdf_folder: str,
    output_dir: Optional[str] = None,
) -> Tuple[List[PaddedHandInfo], List[str], int]:
    """
    Prepare all hands from a folder by padding them to the same joint count.

    Args:
        urdf_folder: Path to folder containing hand URDF subfolders
        output_dir: Directory to save padded URDFs (default: temp directory)

    Returns:
        Tuple of (padded_hands, joint_template, max_joints)
        - padded_hands: List of PaddedHandInfo for each hand
        - joint_template: Sorted list of all joint names (the "universal" joint order)
        - max_joints: Total number of joints after padding
    """
    # Analyze all hands to get max joints, joint template, and canonical tree
    max_joints, joint_template, hand_configs, canonical_tree = analyze_hands(urdf_folder)

    print(f"Found {len(hand_configs)} hands in {urdf_folder}")
    print(f"Max joints: {max_joints}, Total unique joint names: {len(joint_template)}")

    # Use provided output dir or create temp directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="padded_urdfs_")
    else:
        # Always clear old padded URDFs to ensure fresh generation
        import shutil
        output_dir = Path(output_dir)
        if output_dir.exists():
            print(f"  Clearing old padded URDFs from {output_dir}...")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = str(output_dir)

    padded_hands: List[PaddedHandInfo] = []

    for hand_name in sorted(hand_configs.keys()):
        hand_config = hand_configs[hand_name]

        # Pad the URDF using canonical tree topology
        padded_path, real_mask = pad_urdf(
            hand_config.urdf_path,
            joint_template,
            output_dir,
            hand_name,
            canonical_tree=canonical_tree,
        )

        padded_info = PaddedHandInfo(
            original_urdf_path=hand_config.urdf_path,
            padded_urdf_path=padded_path,
            hand_name=hand_name,
            real_joint_mask=real_mask,
            real_joint_names=hand_config.joint_names,
            num_real_joints=hand_config.num_joints,
            num_total_joints=len(joint_template),
            hand_config=hand_config,
        )
        padded_hands.append(padded_info)

        print(f"  {hand_name}: {hand_config.num_joints} joints -> {len(joint_template)} (padded)")

    return padded_hands, joint_template, len(joint_template)


def get_real_joint_indices(real_mask: List[bool]) -> List[int]:
    """
    Get indices of real (non-dummy) joints from a mask.

    Args:
        real_mask: List[bool] where True indicates a real joint

    Returns:
        List of indices where real_mask[i] == True
    """
    return [i for i, is_real in enumerate(real_mask) if is_real]


def extract_real_joint_values(values, real_mask: List[bool]):
    """
    Extract only real joint values from a padded tensor.

    Works with both 1D tensors [num_joints] and 2D tensors [batch, num_joints].

    Args:
        values: Tensor of shape [..., num_joints]
        real_mask: List[bool] where True indicates a real joint

    Returns:
        Tensor with only real joint values: [..., num_real_joints]
    """
    import torch

    real_indices = get_real_joint_indices(real_mask)

    if values.dim() == 1:
        return values[real_indices]
    elif values.dim() == 2:
        return values[:, real_indices]
    else:
        # For higher dimensions, assume last dim is joints
        return values[..., real_indices]


def convert_urdf_to_usd(
    urdf_path: str,
    output_dir: str,
    hand_name: str,
    fix_base: bool = True,
    self_collision: bool = True,
) -> str:
    """
    Convert a URDF file to USD format using Isaac Lab's converter.

    This function must be called after Isaac Sim is launched.

    Args:
        urdf_path: Path to the URDF file
        output_dir: Directory to save the USD file
        hand_name: Name of the hand (for output filename)
        fix_base: Whether to fix the base link
        self_collision: Whether to enable self-collision

    Returns:
        Path to the converted USD file
    """
    from isaaclab.sim.converters import UrdfConverterCfg, UrdfConverter

    # Use absolute paths to avoid sublayer reference issues
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = str(Path(urdf_path).resolve())

    # Configure converter with absolute paths
    # Use same settings as robot_builder.build_robot_cfg
    converter_cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=str(output_dir),
        usd_file_name=f"{hand_name}.usd",
        fix_base=fix_base,
        self_collision=self_collision,
        force_usd_conversion=True,  # Always reconvert
        merge_fixed_joints=False,  # Keep dummy joints separate
        make_instanceable=True,  # Enable instancing for faster scene creation
        collider_type="convex_hull",  # Better collision accuracy (from robot_builder)
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=2.0,
                damping=0.3,
            ),
        ),
    )

    # Run conversion
    converter = UrdfConverter(converter_cfg)

    return converter.usd_path


def convert_all_padded_hands_to_usd(
    padded_hands: List[PaddedHandInfo],
    output_dir: str,
) -> List[PaddedHandInfo]:
    """
    Convert all padded URDF hands to USD format.

    This function must be called after Isaac Sim is launched.

    Args:
        padded_hands: List of PaddedHandInfo with padded URDF paths
        output_dir: Directory to save the USD files

    Returns:
        Updated list of PaddedHandInfo with USD paths set
    """
    import shutil
    output_dir = Path(output_dir)

    # Always clear old USD files to ensure fresh generation
    if output_dir.exists():
        print(f"  Clearing old USD files from {output_dir}...")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Converting {len(padded_hands)} padded URDFs to USD...")

    successful_hands = []
    for hand_info in padded_hands:
        try:
            usd_path = convert_urdf_to_usd(
                urdf_path=hand_info.padded_urdf_path,
                output_dir=str(output_dir),
                hand_name=hand_info.hand_name,
            )
            # Verify the USD file was actually created
            if not Path(usd_path).exists():
                raise RuntimeError(f"USD file not created at {usd_path}")
            # Check file size to ensure it's not empty/corrupt
            if Path(usd_path).stat().st_size < 1000:
                raise RuntimeError(f"USD file appears to be empty or corrupt (size < 1KB)")
            hand_info.usd_path = usd_path
            print(f"  {hand_info.hand_name}: {usd_path} (OK)")
            successful_hands.append(hand_info)
        except Exception as e:
            print(f"  WARNING: Failed to convert {hand_info.hand_name}: {e}")
            print(f"  Skipping this hand...")

    if not successful_hands:
        raise RuntimeError("No hands were successfully converted to USD!")

    if len(successful_hands) < len(padded_hands):
        print(f"\nWARNING: Only {len(successful_hands)}/{len(padded_hands)} hands converted successfully")

    return successful_hands
