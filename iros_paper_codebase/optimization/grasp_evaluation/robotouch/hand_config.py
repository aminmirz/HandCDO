"""
Hand configuration module for parsing URDF and generating hand configurations.
Supports both new generated hands (J_f1_2_4 naming) and legacy LEAP hand (a_0 to a_15).
"""

from __future__ import annotations
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class JointInfo:
    """Information about a single joint in the hand."""
    name: str
    index: int
    finger_id: str         # f1, f2, f3, t1
    position: int          # 1, 2, 3, 4, 5 (position in kinematic chain)
    joint_type: int        # 0-3 = sideways/axial, 4-7 = closing
    is_closing: bool       # True if closing joint (type 4-7), False if sideways (type 0-3)
    lower_limit: float
    upper_limit: float
    closing_direction: int # +1 or -1


@dataclass
class HandConfig:
    """Complete configuration for a robot hand."""
    urdf_path: str
    joint_names: List[str]                      # Sorted list of all joint names
    joint_info: Dict[str, JointInfo]            # name -> JointInfo
    joints_by_finger: Dict[str, List[int]]      # finger_id -> [indices]
    joints_by_position: Dict[int, List[int]]    # position -> [indices]
    finger_order: List[List[int]]               # Groups for wave closing
    closing_directions: List[int]               # Per-joint closing direction
    num_joints: int
    num_fingers: int
    max_joint_position: int                     # Max joints per finger
    base_link_name: str = "base_link"

    # Base rotation quaternion (w, x, y, z) to orient palm facing +Z
    # URDF hands need 90° rotation around X, LEAP hands use identity
    base_rotation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # For backward compatibility with LEAP hand specific code
    spread_joint_indices: List[int] = field(default_factory=list)  # Joints at position 1 (spread/rotation)

    # Joint type categorization (for URDF hands with J_f{n}_{m}_{k} naming)
    sideways_joint_indices: List[int] = field(default_factory=list)  # Joints with k=0,1,2,3 (sideways/axial)
    closing_joint_indices: List[int] = field(default_factory=list)   # Joints with k=4,5,6,7 (closing)
    sideways_order: List[List[int]] = field(default_factory=list)    # Groups for per-level sideways (by k=0,1,2,3)

    # Fanning vs Axial categorization (based on position in kinematic chain)
    fanning_joint_indices: List[int] = field(default_factory=list)   # Sideways joints at position 1 (base spreading)
    axial_joint_indices: List[int] = field(default_factory=list)     # Sideways joints at other positions (rotation)


def parse_joint_name(name: str) -> Optional[Tuple[str, int, int]]:
    """
    Parse joint name to extract finger_id, position, and joint_type.

    Supports two naming conventions:
    - New URDF format: J_f1_2_4, J_t1_3_7 -> (f1, 2, 4), (t1, 3, 7)
    - LEAP format: a_0, a_1, ... a_15 -> requires manual mapping

    Joint type (k value):
    - 0,1,2,3 = sideways/axial rotation joints
    - 4,5,6,7 = closing joints

    Returns:
        Tuple of (finger_id, position, joint_type) or None if name doesn't match patterns
    """
    # New URDF format: J_{finger}_{position}_{type}
    pattern_new = r'^J_(f\d|t\d)_(\d+)_(\d+)$'
    match = re.match(pattern_new, name)
    if match:
        finger_id = match.group(1)
        position = int(match.group(2))
        joint_type = int(match.group(3))
        return (finger_id, position, joint_type)

    return None


def get_closing_direction(lower_limit: float, upper_limit: float) -> int:
    """
    Determine closing direction based on joint limits.

    Assumes joints close toward the limit that is further from zero,
    which typically corresponds to finger flexion.

    Returns:
        +1 if closing by incrementing, -1 if closing by decrementing
    """
    # If upper limit magnitude is greater, close toward upper (positive direction)
    if abs(upper_limit) > abs(lower_limit):
        return +1
    else:
        return -1


def build_finger_order(joints: List[JointInfo], joint_to_idx: Dict[str, int], max_position_gap: int = 3) -> List[List[int]]:
    """
    Group CLOSING joints by their ORDER within each finger for synchronized wave closing.

    Each finger's 1st closing joint closes together, 2nd closes together, etc.
    This synchronizes fingers even if they have different starting positions.

    For example:
    - Finger 1: closing joints at positions 2, 3, 4, 5 → 1st, 2nd, 3rd, 4th
    - Finger 2: closing joints at positions 1, 2, 3, 4 → 1st, 2nd, 3rd, 4th
    Wave 1 groups: f1_pos2 + f2_pos1 (both are "1st closing joint")

    Args:
        joints: List of JointInfo objects
        joint_to_idx: Mapping from joint name to index
        max_position_gap: Maximum position difference to allow in same wave (default 3)

    Returns:
        finger_order: list of wave groups for synchronized closing
    """
    # First, organize closing joints by finger, sorted by position
    finger_closing_joints: Dict[str, List[JointInfo]] = {}

    for j in joints:
        if not j.is_closing:
            continue
        if j.finger_id not in finger_closing_joints:
            finger_closing_joints[j.finger_id] = []
        finger_closing_joints[j.finger_id].append(j)

    # Sort each finger's joints by position (base to tip)
    for finger_id in finger_closing_joints:
        finger_closing_joints[finger_id].sort(key=lambda x: x.position)

    if not finger_closing_joints:
        return []

    # Find max number of closing joints any finger has
    max_joints_per_finger = max(len(joints_list) for joints_list in finger_closing_joints.values())

    # Build waves: wave N contains the Nth closing joint from each finger
    finger_order = []
    for wave_idx in range(max_joints_per_finger):
        wave_joints = []
        positions_in_wave = []

        for finger_id, joints_list in finger_closing_joints.items():
            if wave_idx < len(joints_list):
                j = joints_list[wave_idx]
                wave_joints.append((joint_to_idx[j.name], j.position))
                positions_in_wave.append(j.position)

        if not wave_joints:
            continue

        # Check position gap - if joints are too far apart, split into sub-waves
        if max_position_gap > 0 and len(positions_in_wave) > 1:
            min_pos = min(positions_in_wave)
            max_pos = max(positions_in_wave)

            if max_pos - min_pos > max_position_gap:
                # Split: group joints that are within max_position_gap of each other
                # Sort by position and create sub-groups
                wave_joints.sort(key=lambda x: x[1])
                sub_wave = [wave_joints[0][0]]
                sub_wave_base_pos = wave_joints[0][1]

                for idx, pos in wave_joints[1:]:
                    if pos - sub_wave_base_pos <= max_position_gap:
                        sub_wave.append(idx)
                    else:
                        # Start new sub-wave
                        finger_order.append(sorted(sub_wave))
                        sub_wave = [idx]
                        sub_wave_base_pos = pos

                if sub_wave:
                    finger_order.append(sorted(sub_wave))
            else:
                # All joints within gap, add as single wave
                finger_order.append(sorted([idx for idx, pos in wave_joints]))
        else:
            finger_order.append(sorted([idx for idx, pos in wave_joints]))

    return finger_order


def build_sideways_order(joints: List[JointInfo], joint_to_idx: Dict[str, int]) -> List[List[int]]:
    """
    Group SIDEWAYS joints by their joint_type (k value) for ordered actuation.

    Only includes sideways joints (is_closing=False, joint_type 0-3).

    Joint type k determines actuation order:
    - k=0 actuates first
    - k=1 actuates second
    - k=2 actuates third
    - k=3 actuates last

    Returns:
        sideways_order: list of groups, where each group contains
        indices of sideways joints at that order level.
    """
    # Group sideways joints by joint_type (k value determines order)
    type_groups: Dict[int, List[int]] = {}

    for j in joints:
        # Only include sideways joints
        if j.is_closing:
            continue

        k = j.joint_type  # k=0,1,2,3 determines actuation order
        if k not in type_groups:
            type_groups[k] = []
        type_groups[k].append(joint_to_idx[j.name])

    # Sort by joint_type (0, 1, 2, 3) and build sideways_order
    if not type_groups:
        return []

    sideways_order = []
    for k in sorted(type_groups.keys()):
        # Sort indices within each group for consistency
        sideways_order.append(sorted(type_groups[k]))

    return sideways_order


def parse_urdf(urdf_path: str) -> HandConfig:
    """
    Parse a URDF file to extract hand configuration.

    Args:
        urdf_path: Path to the URDF file

    Returns:
        HandConfig with all joint information extracted
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Find all revolute joints
    joints_data = []
    for joint in root.findall('joint'):
        joint_type = joint.get('type')
        if joint_type != 'revolute':
            continue

        name = joint.get('name')
        parsed = parse_joint_name(name)
        if parsed is None:
            continue

        finger_id, position, joint_type = parsed
        is_closing = joint_type >= 4  # k=4,5,6,7 are closing joints

        # Get joint limits
        limit = joint.find('limit')
        if limit is not None:
            lower = float(limit.get('lower', '0'))
            upper = float(limit.get('upper', '0'))
        else:
            lower, upper = -1.57, 1.57  # Default to ~90 degrees

        joints_data.append({
            'name': name,
            'finger_id': finger_id,
            'position': position,
            'joint_type': joint_type,
            'is_closing': is_closing,
            'lower': lower,
            'upper': upper,
        })

    # Sort joints alphabetically for consistent indexing
    joints_data.sort(key=lambda x: x['name'])

    # Build data structures
    joint_names = [j['name'] for j in joints_data]
    joint_to_idx = {name: i for i, name in enumerate(joint_names)}

    joint_info = {}
    joints_by_finger: Dict[str, List[int]] = {}
    joints_by_position: Dict[int, List[int]] = {}
    closing_directions = []
    sideways_joint_indices = []
    closing_joint_indices = []
    fanning_joint_indices = []   # First sideways joint per finger (base spreading)
    axial_joint_indices = []     # Remaining sideways joints
    sideways_by_finger: Dict[str, List[tuple]] = {}

    joints_list = []
    for i, jd in enumerate(joints_data):
        closing_dir = get_closing_direction(jd['lower'], jd['upper'])
        closing_directions.append(closing_dir)

        ji = JointInfo(
            name=jd['name'],
            index=i,
            finger_id=jd['finger_id'],
            position=jd['position'],
            joint_type=jd['joint_type'],
            is_closing=jd['is_closing'],
            lower_limit=jd['lower'],
            upper_limit=jd['upper'],
            closing_direction=closing_dir,
        )
        joint_info[jd['name']] = ji
        joints_list.append(ji)

        # Categorize by joint type (k value)
        if jd['is_closing']:
            closing_joint_indices.append(i)
        else:
            sideways_joint_indices.append(i)
            if jd['finger_id'] not in sideways_by_finger:
                sideways_by_finger[jd['finger_id']] = []
            sideways_by_finger[jd['finger_id']].append((i, jd['position']))

        # Group by finger
        if jd['finger_id'] not in joints_by_finger:
            joints_by_finger[jd['finger_id']] = []
        joints_by_finger[jd['finger_id']].append(i)

        # Group by position
        if jd['position'] not in joints_by_position:
            joints_by_position[jd['position']] = []
        joints_by_position[jd['position']].append(i)

    # Classify sideways joints: first (lowest position) = fanning, rest = axial
    for finger_id, joints in sideways_by_finger.items():
        joints.sort(key=lambda x: x[1])
        if not joints:
            continue
        min_pos = joints[0][1]
        for joint_idx, pos in joints:
            if pos == min_pos:
                fanning_joint_indices.append(joint_idx)
            else:
                axial_joint_indices.append(joint_idx)

    # Build finger order for wave closing (only closing joints)
    finger_order = build_finger_order(joints_list, joint_to_idx)

    # Build sideways order for per-level fanning (only sideways joints)
    sideways_order = build_sideways_order(joints_list, joint_to_idx)

    # Calculate metadata
    num_joints = len(joint_names)
    num_fingers = len(joints_by_finger)
    max_position = max(joints_by_position.keys()) if joints_by_position else 0

    # Spread joints are at position 1 (legacy, for LEAP compatibility)
    spread_joint_indices = joints_by_position.get(1, [])

    # Find base link name
    base_link_name = "base_link"
    for link in root.findall('link'):
        link_name = link.get('name')
        if link_name and ('base' in link_name.lower() or 'palm' in link_name.lower()):
            base_link_name = link_name
            break

    # URDF hands need -90° rotation around X (palm -Y to +Z), then +90° around Z
    # Combined quaternion: q_z * q_x where q_x=(-90° X), q_z=(+90° Z)
    # Result: (0.5, -0.5, -0.5, 0.5)
    base_rotation = (0.5, -0.5, -0.5, 0.5)

    return HandConfig(
        urdf_path=urdf_path,
        joint_names=joint_names,
        joint_info=joint_info,
        joints_by_finger=joints_by_finger,
        joints_by_position=joints_by_position,
        finger_order=finger_order,
        closing_directions=closing_directions,
        num_joints=num_joints,
        num_fingers=num_fingers,
        max_joint_position=max_position,
        base_link_name=base_link_name,
        base_rotation=base_rotation,
        spread_joint_indices=spread_joint_indices,
        sideways_joint_indices=sideways_joint_indices,
        closing_joint_indices=closing_joint_indices,
        sideways_order=sideways_order,
        fanning_joint_indices=fanning_joint_indices,
        axial_joint_indices=axial_joint_indices,
    )


def get_leap_hand_config() -> HandConfig:
    """
    Create a HandConfig for the LEAP hand with hardcoded joint mapping.

    LEAP hand has 16 joints (a_0 to a_15) organized as:
    - Index finger: a_0(PIP), a_1(MCP), a_2(DIP), a_3(TIP)
    - Middle finger: a_4(PIP), a_5(MCP), a_6(DIP), a_7(TIP)
    - Ring finger: a_8(PIP), a_9(MCP), a_10(DIP), a_11(TIP)
    - Thumb: a_12(ABD), a_13(PIP), a_14(DIP), a_15(TIP)

    Returns:
        HandConfig for LEAP hand
    """
    from robotouch.paths import LEAP_HAND_USD
    urdf_path = str(LEAP_HAND_USD)

    # LEAP hand joint names in order
    joint_names = [f"a_{i}" for i in range(16)]

    # Manual mapping of LEAP joints to finger_id and position
    # Joint mappings: (finger_id, position)
    # Position 1 = MCP/ABD (base), Position 2 = PIP, Position 3 = DIP, Position 4 = TIP
    leap_mapping = {
        'a_0': ('f1', 2),   # Index PIP
        'a_1': ('f1', 1),   # Index MCP (spread)
        'a_2': ('f1', 3),   # Index DIP
        'a_3': ('f1', 4),   # Index TIP
        'a_4': ('f2', 2),   # Middle PIP
        'a_5': ('f2', 1),   # Middle MCP (spread)
        'a_6': ('f2', 3),   # Middle DIP
        'a_7': ('f2', 4),   # Middle TIP
        'a_8': ('f3', 2),   # Ring PIP
        'a_9': ('f3', 1),   # Ring MCP (spread)
        'a_10': ('f3', 3),  # Ring DIP
        'a_11': ('f3', 4),  # Ring TIP
        'a_12': ('t1', 1),  # Thumb ABD (spread)
        'a_13': ('t1', 2),  # Thumb PIP
        'a_14': ('t1', 3),  # Thumb DIP
        'a_15': ('t1', 4),  # Thumb TIP
    }

    # Joint limits for LEAP hand (approximate)
    leap_limits = {
        'a_0': (-0.314, 2.23),
        'a_1': (-0.105, 1.047),
        'a_2': (-0.506, 1.885),
        'a_3': (-0.366, 2.042),
        'a_4': (-0.314, 2.23),
        'a_5': (-0.105, 1.047),
        'a_6': (-0.506, 1.885),
        'a_7': (-0.366, 2.042),
        'a_8': (-0.314, 2.23),
        'a_9': (-0.105, 1.047),
        'a_10': (-0.506, 1.885),
        'a_11': (-0.366, 2.042),
        'a_12': (0.263, 2.042),     # Thumb ABD
        'a_13': (-0.209, 2.042),
        'a_14': (-1.571, 1.571),    # Thumb DIP has different range
        'a_15': (-0.209, 2.042),
    }

    joint_to_idx = {name: i for i, name in enumerate(joint_names)}

    joint_info = {}
    joints_by_finger: Dict[str, List[int]] = {}
    joints_by_position: Dict[int, List[int]] = {}
    closing_directions = []
    sideways_joint_indices = []
    closing_joint_indices = []
    fanning_joint_indices = []   # LEAP: all sideways are at position 1 (fanning)
    axial_joint_indices = []     # LEAP: no axial joints
    joints_list = []

    for i, name in enumerate(joint_names):
        finger_id, position = leap_mapping[name]
        lower, upper = leap_limits[name]
        closing_dir = get_closing_direction(lower, upper)
        closing_directions.append(closing_dir)

        # For LEAP: position 1 = sideways (MCP/ABD), positions 2-4 = closing
        is_closing = position != 1
        joint_type = 4 if is_closing else 0  # Use 0 for sideways, 4 for closing (to match URDF convention)

        ji = JointInfo(
            name=name,
            index=i,
            finger_id=finger_id,
            position=position,
            joint_type=joint_type,
            is_closing=is_closing,
            lower_limit=lower,
            upper_limit=upper,
            closing_direction=closing_dir,
        )
        joint_info[name] = ji
        joints_list.append(ji)

        # Categorize by joint type
        if is_closing:
            closing_joint_indices.append(i)
        else:
            sideways_joint_indices.append(i)
            # LEAP: all sideways joints are at position 1 (fanning)
            fanning_joint_indices.append(i)

        # Group by finger
        if finger_id not in joints_by_finger:
            joints_by_finger[finger_id] = []
        joints_by_finger[finger_id].append(i)

        # Group by position
        if position not in joints_by_position:
            joints_by_position[position] = []
        joints_by_position[position].append(i)

    # Build finger order for wave closing (only closing joints, by position)
    finger_order = build_finger_order(joints_list, joint_to_idx)

    # Build sideways order for per-level fanning (for LEAP, all in one group)
    sideways_order = build_sideways_order(joints_list, joint_to_idx)

    # Spread joints are at position 1 (same as sideways for LEAP)
    spread_joint_indices = joints_by_position.get(1, [])

    return HandConfig(
        urdf_path=urdf_path,
        joint_names=joint_names,
        joint_info=joint_info,
        joints_by_finger=joints_by_finger,
        joints_by_position=joints_by_position,
        finger_order=finger_order,
        closing_directions=closing_directions,
        num_joints=16,
        num_fingers=4,
        max_joint_position=4,
        base_link_name="base",
        spread_joint_indices=spread_joint_indices,
        sideways_joint_indices=sideways_joint_indices,
        closing_joint_indices=closing_joint_indices,
        sideways_order=sideways_order,
        fanning_joint_indices=fanning_joint_indices,
        axial_joint_indices=axial_joint_indices,
    )


def get_hand_config(hand_type: str = "leap", urdf_path: str = "") -> HandConfig:
    """
    Get hand configuration based on type.

    Args:
        hand_type: "leap" for LEAP hand, "urdf" for custom URDF
        urdf_path: Path to URDF file (required if hand_type is "urdf")

    Returns:
        HandConfig for the specified hand
    """
    if hand_type == "leap":
        return get_leap_hand_config()
    elif hand_type == "urdf":
        if not urdf_path:
            raise ValueError("urdf_path is required when hand_type is 'urdf'")
        return parse_urdf(urdf_path)
    else:
        raise ValueError(f"Unknown hand_type: {hand_type}")
