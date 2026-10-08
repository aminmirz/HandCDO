bl_info = {
    "name": "Hand Generator V2",
    "author": "Hand Generation Team",
    "version": (2, 0, 0),
    "blender": (4, 5, 0),
    "location": "View3D > Sidebar > Hand Gen V2",
    "description": "Parametric robotic hand generator using generation_v2",
    "category": "Object",
}

import bpy
import sys
import os
import importlib
import importlib.util
import tempfile
import shutil
import random
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ADDON_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_generation_v2_dir():
    """Locate generation_v2 whether the add-on runs from the repo or Blender's addon folder."""
    candidates = [
        os.path.abspath(os.path.join(ADDON_DIR, os.pardir)),
        os.path.abspath(os.path.join(ADDON_DIR, "generation_v2")),
        r"C:\Users\aminm\Desktop\HandGeneration\generation_v2",
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "Config.py")) and os.path.isfile(os.path.join(path, "blender_full_assembly.py")):
            return path
    return candidates[-1]


GENERATION_DIR = _find_generation_v2_dir()
COMPONENTS_BLEND = os.path.join(GENERATION_DIR, "blender", "components.blend")


def _is_valid_generation_v2_dir(path):
    path = bpy.path.abspath(path) if path else ""
    path = os.path.abspath(path) if path else ""
    required = ("Config.py", "HandClass.py", "blender_full_assembly.py")
    return bool(path) and all(os.path.isfile(os.path.join(path, name)) for name in required)


def _set_generation_v2_dir(path):
    """Update global generation_v2 paths from the add-on setting."""
    global GENERATION_DIR, COMPONENTS_BLEND
    resolved = bpy.path.abspath(path) if path else ""
    resolved = os.path.abspath(resolved) if resolved else ""
    if not _is_valid_generation_v2_dir(resolved):
        raise FileNotFoundError(
            "Generation V2 folder must contain Config.py, HandClass.py, and blender_full_assembly.py"
        )
    old_path = GENERATION_DIR
    GENERATION_DIR = resolved
    COMPONENTS_BLEND = os.path.join(GENERATION_DIR, "blender", "components.blend")
    if old_path in sys.path and old_path != GENERATION_DIR:
        try:
            sys.path.remove(old_path)
        except ValueError:
            pass
    _ensure_gen_path()
    return GENERATION_DIR


def _ensure_gen_path():
    if GENERATION_DIR not in sys.path:
        sys.path.insert(0, GENERATION_DIR)


def _ensure_matplotlib_mock():
    """Install a lightweight mock for matplotlib if it is not available.

    The generation modules import matplotlib at the top level for plotting,
    but the plots are never produced during addon use. Providing a mock
    prevents ImportError inside Blender's Python (which usually lacks matplotlib).
    """
    try:
        import matplotlib  # noqa: F401
        return  # real matplotlib is available
    except ImportError:
        pass

    import types

    def _noop(*a, **kw):
        return _noop

    # Minimal mock that returns itself for any attribute access or call
    class _Mock:
        def __init__(self, *a, **kw):
            pass
        def __getattr__(self, name):
            return _noop
        def __call__(self, *a, **kw):
            return self

    mock = types.ModuleType("matplotlib")
    mock.__path__ = []
    mock.__file__ = ""
    sys.modules["matplotlib"] = mock
    sys.modules["matplotlib.pyplot"] = _Mock()
    sys.modules["matplotlib.patches"] = _Mock()
    sys.modules["matplotlib.cm"] = _Mock()
    sys.modules["matplotlib.colors"] = _Mock()
    sys.modules["matplotlib.collections"] = _Mock()
    sys.modules["mpl_toolkits"] = types.ModuleType("mpl_toolkits")
    sys.modules["mpl_toolkits"].__path__ = []
    sys.modules["mpl_toolkits.mplot3d"] = types.ModuleType("mpl_toolkits.mplot3d")
    sys.modules["mpl_toolkits.mplot3d"].__path__ = []
    mplot3d_art3d = _Mock()
    mplot3d_art3d.Poly3DCollection = _Mock
    sys.modules["mpl_toolkits.mplot3d.art3d"] = mplot3d_art3d


# ---------------------------------------------------------------------------
# Property group
# ---------------------------------------------------------------------------

def _prop_update(self, context):
    """Debounced auto-update when a property changes."""
    if not context.scene.handgen_settings.auto_update:
        return
    global _auto_update_timer
    if _auto_update_timer is not None:
        try:
            bpy.app.timers.unregister(_auto_update_timer)
        except Exception:
            pass
    _auto_update_timer = _auto_update_cb
    bpy.app.timers.register(_auto_update_cb, first_interval=0.5)


def _visibility_update(self, context):
    """Apply visibility-only settings to the already generated hand."""
    try:
        _apply_hand_visibility(context)
    except Exception:
        import traceback
        traceback.print_exc()


def _overlay_update(self, context):
    """Mirror Blender's viewport overlay toggle across all 3D views."""
    show = bool(context.scene.handgen_settings.show_overlays)
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.overlay.show_overlays = show


class HANDGEN_PG_Settings(bpy.types.PropertyGroup):

    generation_mode: bpy.props.EnumProperty(
        name="Generation Mode",
        items=[
            ('standard', 'Standard', 'Use the normal finger/thumb generation mode'),
            ('finger_only', 'Finger Only', 'Use only finger structures placed around the full palm outline'),
        ],
        default='standard',
        update=_prop_update,
    )

    # --- Palm shape ---
    palm_size_mm: bpy.props.FloatProperty(
        name="Palm Size (mm)", default=168.0, min=60.0, max=400.0,
        description="Final clean outer palm X width in mm", update=_prop_update,
    )
    outline_sides: bpy.props.IntProperty(
        name="Outline Sides", default=4, min=3, max=20,
        description="Number of polygon sides for the palm outline", update=_prop_update,
    )
    outline_aspect_ratio: bpy.props.FloatProperty(
        name="Aspect Ratio", default=1.55, min=0.5, max=3.0,
        description="Aspect ratio of the palm outline", update=_prop_update,
    )
    outline_longer_axis: bpy.props.EnumProperty(
        name="Longer Axis",
        items=[('x', 'X', ''), ('y', 'Y', '')],
        default='x', update=_prop_update,
    )
    outline_rotation_deg: bpy.props.FloatProperty(
        name="Rotation (deg)", default=0.0, min=0.0, max=180.0, update=_prop_update,
    )
    smoothing_iters: bpy.props.IntProperty(
        name="Smoothing Iters", default=6, min=0, max=15, update=_prop_update,
    )
    smoothing_t: bpy.props.FloatProperty(
        name="Smoothing T", default=0.25, min=0.0, max=1.0, update=_prop_update,
    )
    resolution_mm: bpy.props.FloatProperty(
        name="Resolution (mm)", default=3.0, min=1.0, max=20.0, update=_prop_update,
    )

    # --- Fingers / thumbs ---
    finger_number: bpy.props.IntProperty(
        name="Fingers", default=3, min=0, max=8, update=_prop_update,
    )
    thumb_number: bpy.props.IntProperty(
        name="Thumbs", default=1, min=0, max=3, update=_prop_update,
    )
    thumb_side: bpy.props.EnumProperty(
        name="Thumb Side",
        items=[('right', 'Right', ''), ('left', 'Left', ''), ('random', 'Random', '')],
        default='right', update=_prop_update,
    )
    # --- Hand geometry ---
    palm_extrude_height_mm: bpy.props.FloatProperty(
        name="Palm Height (mm)", default=45.0, min=10.0, max=100.0, update=_prop_update,
    )
    palm_thickness_mm: bpy.props.FloatProperty(
        name="Palm Thickness (mm)", default=4.0, min=1.0, max=15.0, update=_prop_update,
    )
    palm_wall_thickness_mm: bpy.props.FloatProperty(
        name="Wall Thickness (mm)", default=4.0, min=1.0, max=15.0, update=_prop_update,
    )
    finger_palm_base_offset_mm: bpy.props.FloatProperty(
        name="Finger Base Offset (mm)", default=10.0, min=0.0, max=30.0, update=_prop_update,
    )

    # --- Options ---
    detailed_viz: bpy.props.BoolProperty(
        name="Detailed Viz", default=False, update=_prop_update,
    )
    generate_collision_mesh: bpy.props.BoolProperty(
        name="Make Collision Mesh", default=True, update=_prop_update,
        description="Generate collision mesh objects when the hand is rebuilt",
    )
    show_collision_mesh: bpy.props.BoolProperty(
        name="Show Collision Mesh", default=False, update=_visibility_update,
        description="Show or hide existing collision mesh objects without regenerating",
    )
    show_revolute: bpy.props.BoolProperty(
        name="Show Revolute", default=False, update=_visibility_update,
    )
    show_viz: bpy.props.BoolProperty(
        name="Show Viz", default=True, update=_visibility_update,
    )
    show_attach: bpy.props.BoolProperty(
        name="Show Attach", default=False, update=_visibility_update,
    )
    show_fingers: bpy.props.BoolProperty(
        name="Show Fingers", default=True, update=_visibility_update,
    )
    show_thumbs: bpy.props.BoolProperty(
        name="Show Thumbs", default=True, update=_visibility_update,
    )
    show_palm: bpy.props.BoolProperty(
        name="Show Palm", default=True, update=_visibility_update,
    )
    show_overlays: bpy.props.BoolProperty(
        name="Show Overlays",
        default=False,
        description="Show or hide Blender viewport overlays in all 3D views",
        update=_overlay_update,
    )

    # --- Pad (kernel) settings ---
    show_pad_settings: bpy.props.BoolProperty(
        name="Palm Pad", default=False, update=_prop_update,
    )
    pad_resolution: bpy.props.IntProperty(
        name="Pad Resolution", default=4, min=1, max=10,
        description="Resolution level of the palm pad surface", update=_prop_update,
    )
    pad_thickness_mm: bpy.props.FloatProperty(
        name="Pad Thickness (mm)", default=1.0, min=0.1, max=10.0,
        description="Palm pad extrusion thickness in mm", update=_prop_update,
    )
    pad_max_intensity: bpy.props.FloatProperty(
        name="Max Intensity (mm)", default=0.0,
        description="Maximum bump height in mm", update=_prop_update,
    )
    pad_kernel_count: bpy.props.IntProperty(
        name="Kernels", default=0, min=0, max=5, update=_prop_update,
    )

    copy_finger_settings: bpy.props.BoolProperty(
        name="Copy to Other Fingers", default=True, update=_prop_update,
    )
    copy_thumb_settings: bpy.props.BoolProperty(
        name="Copy to Other Thumbs", default=True, update=_prop_update,
    )
    use_custom_locations: bpy.props.BoolProperty(
        name="Use Custom t Locations",
        default=False,
        description="Edit per-finger/thumb location t on the valid outline curve",
        update=_prop_update,
    )

    # Per-finger and per-thumb properties are added programmatically below.

    # --- Misc ---
    auto_update: bpy.props.BoolProperty(name="Auto Update", default=False)
    generation_v2_dir: bpy.props.StringProperty(
        name="Generation V2 Folder",
        subtype='DIR_PATH',
        default=GENERATION_DIR,
        description="Folder containing generation_v2 Config.py and blender_full_assembly.py",
    )
    save_output_dir: bpy.props.StringProperty(
        name="Save Folder",
        subtype='DIR_PATH',
        default=r"C:\Users\aminm\Desktop\HandGeneration\generation_v2\generated_hands\blender_addon",
        description="Folder where saved hands are written",
    )
    save_hand_name: bpy.props.StringProperty(
        name="Hand Name",
        default="hand_addon",
        description="Subfolder name for the saved hand",
    )


# ---------------------------------------------------------------------------
# Per-finger / per-thumb properties (added programmatically)
# ---------------------------------------------------------------------------

_FINGER_ROT_ITEMS = [
    ('0', 'None', 'No rotation'),
    ('3', 'Vert', 'Vertical axis rotation'),
    ('1', 'H-Short', 'Horizontal axis rotation - short'),
    ('2', 'H-Long', 'Horizontal axis rotation - long'),
    ('4', 'V+H', 'Vertical + horizontal rotation'),
]
_THUMB_ROT_ITEMS = [
    ('0', 'Primary', 'Primary rotation only'),
    ('1', 'Pri+Sec', 'Primary + secondary rotation'),
]

for _i in range(8):  # max 8 fingers
    _finger_location_defaults = tuple((i + 1) / 9.0 for i in range(8))
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_location'] = bpy.props.FloatProperty(
        name="Location t",
        default=_finger_location_defaults[_i],
        min=0.0,
        max=1.0,
        description="Finger location on the valid outline curve, 0=right, 1=left",
        update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_rotation'] = bpy.props.EnumProperty(
        name="Rot", items=_FINGER_ROT_ITEMS, default='2', update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_before'] = bpy.props.StringProperty(
        name="Before", default="1",
        description="Before-rotation joints (1=short, 2=long)", update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_after'] = bpy.props.StringProperty(
        name="After", default="1",
        description="After-rotation joints (1=short, 2=long)", update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_angle'] = bpy.props.FloatProperty(
        name="Angle (deg)", default=0.0,
        description="Rotation angle of the finger base in degrees", update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_normal_offset'] = bpy.props.FloatProperty(
        name="Normal Offset", default=0.0, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_side_offset'] = bpy.props.FloatProperty(
        name="Side Offset",
        default=0.0,
        description="Signed movement along the valid outline curve in mm; positive moves left",
        update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_fingertip_scale'] = bpy.props.FloatVectorProperty(
        name="Fingertip Size", default=(1.0, 1.0, 1.0), size=3, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_link_added_length'] = bpy.props.FloatProperty(
        name="Added Link Length (mm)", default=0.0, min=0.0,
        description="Extra length added to each link (mm)", update=_prop_update)

for _i in range(3):  # max 3 thumbs
    _thumb_location_defaults = (0.5, 0.75, 0.9)
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_location'] = bpy.props.FloatProperty(
        name="Location t",
        default=_thumb_location_defaults[_i],
        min=0.0,
        max=1.0,
        description="Thumb location on the valid outline curve, 0=right, 1=left",
        update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_rotation'] = bpy.props.EnumProperty(
        name="Rot", items=_THUMB_ROT_ITEMS, default='1', update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_after'] = bpy.props.StringProperty(
        name="Joints", default="22",
        description="Joints after rotation (1=short, 2=long)", update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_angle'] = bpy.props.FloatProperty(
        name="Angle (deg)", default=0.0,
        description="Rotation angle of the thumb base in degrees", update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_normal_offset'] = bpy.props.FloatProperty(
        name="Normal Offset", default=0.0, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_side_offset'] = bpy.props.FloatProperty(
        name="Side Offset",
        default=0.0,
        description="Signed movement along the valid outline curve in mm; positive moves left",
        update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_fingertip_scale'] = bpy.props.FloatVectorProperty(
        name="Fingertip Size", default=(1.0, 1.0, 1.0), size=3, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_link_added_length'] = bpy.props.FloatProperty(
        name="Added Link Length (mm)", default=0.0, min=0.0,
        description="Extra length added to each link (mm)", update=_prop_update)

for _i in range(5):  # max 5 kernels
    HANDGEN_PG_Settings.__annotations__[f'pad_kernel_{_i}_aspect_ratio'] = bpy.props.FloatProperty(
        name="Aspect Ratio", default=1.0, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'pad_kernel_{_i}_spread'] = bpy.props.FloatProperty(
        name="Spread", default=0.25, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'pad_kernel_{_i}_rotation'] = bpy.props.FloatProperty(
        name="Rotation (deg)", default=0.0, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'pad_kernel_{_i}_intensity'] = bpy.props.FloatProperty(
        name="Intensity Ratio", default=1.0, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'pad_kernel_{_i}_center_offset'] = bpy.props.FloatProperty(
        name="Center Offset", default=0.0, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'pad_kernel_{_i}_center_angle'] = bpy.props.FloatProperty(
        name="Center Angle (deg)", default=0.0, update=_prop_update)

# Per-finger pad properties (max 8 fingers, max 3 kernels each)
for _i in range(8):
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_show_pad'] = bpy.props.BoolProperty(
        name="Joint Pads", default=False, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_pad_resolution'] = bpy.props.IntProperty(
        name="Pad Resolution", default=2, min=1, max=10, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_pad_thickness'] = bpy.props.FloatProperty(
        name="Pad Thickness (mm)", default=1.0, min=0.1, max=10.0, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_pad_max_intensity'] = bpy.props.FloatProperty(
        name="Max Intensity (mm)", default=0.0, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_pad_kernel_count'] = bpy.props.IntProperty(
        name="Kernels", default=0, min=0, max=3, update=_prop_update)
    for _j in range(3):
        HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_pad_k{_j}_intensity'] = bpy.props.FloatProperty(
            name="Intensity Ratio", default=1.0, update=_prop_update)
        HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_pad_k{_j}_spread'] = bpy.props.FloatProperty(
            name="Spread", default=0.25, update=_prop_update)
        HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_pad_k{_j}_aspect_ratio'] = bpy.props.FloatProperty(
            name="Aspect Ratio", default=1.0, update=_prop_update)
        HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_pad_k{_j}_rotation'] = bpy.props.FloatProperty(
            name="Rotation (deg)", default=0.0, update=_prop_update)
        HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_pad_k{_j}_center_offset'] = bpy.props.FloatProperty(
            name="Center Offset", default=0.0, update=_prop_update)
        HANDGEN_PG_Settings.__annotations__[f'finger_{_i}_pad_k{_j}_center_angle'] = bpy.props.FloatProperty(
            name="Center Angle (deg)", default=0.0, update=_prop_update)

# Per-thumb pad properties (max 3 thumbs, max 3 kernels each)
for _i in range(3):
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_show_pad'] = bpy.props.BoolProperty(
        name="Joint Pads", default=False, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_pad_resolution'] = bpy.props.IntProperty(
        name="Pad Resolution", default=2, min=1, max=10, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_pad_thickness'] = bpy.props.FloatProperty(
        name="Pad Thickness (mm)", default=1.0, min=0.1, max=10.0, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_pad_max_intensity'] = bpy.props.FloatProperty(
        name="Max Intensity (mm)", default=0.0, update=_prop_update)
    HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_pad_kernel_count'] = bpy.props.IntProperty(
        name="Kernels", default=0, min=0, max=3, update=_prop_update)
    for _j in range(3):
        HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_pad_k{_j}_intensity'] = bpy.props.FloatProperty(
            name="Intensity Ratio", default=1.0, update=_prop_update)
        HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_pad_k{_j}_spread'] = bpy.props.FloatProperty(
            name="Spread", default=0.25, update=_prop_update)
        HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_pad_k{_j}_aspect_ratio'] = bpy.props.FloatProperty(
            name="Aspect Ratio", default=1.0, update=_prop_update)
        HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_pad_k{_j}_rotation'] = bpy.props.FloatProperty(
            name="Rotation (deg)", default=0.0, update=_prop_update)
        HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_pad_k{_j}_center_offset'] = bpy.props.FloatProperty(
            name="Center Offset", default=0.0, update=_prop_update)
        HANDGEN_PG_Settings.__annotations__[f'thumb_{_i}_pad_k{_j}_center_angle'] = bpy.props.FloatProperty(
            name="Center Angle (deg)", default=0.0, update=_prop_update)


# ---------------------------------------------------------------------------
# Code builder helpers
# ---------------------------------------------------------------------------

def _encode_joints(joint_str):
    """Convert a string of 1s and 2s to joint type pairs (1→01, 2→10)."""
    out = ""
    for c in joint_str:
        out += "01" if c == "1" else "10"
    return out


def _build_finger_code(settings, i):
    rotation = getattr(settings, f"finger_{i}_rotation")
    before = ''.join(c for c in getattr(settings, f"finger_{i}_before") if c in "12")
    after = ''.join(c for c in getattr(settings, f"finger_{i}_after") if c in "12")
    if before == "" and after == "":
        after = "1"
    # Pad to 5 pairs (10 chars) – max_add_joint for fingers is 5
    return f"{rotation}-{before}-{after}"


def _build_thumb_code(settings, i):
    rotation = getattr(settings, f"thumb_{i}_rotation")
    after = ''.join(c for c in getattr(settings, f"thumb_{i}_after") if c in "12")
    if after == "":
        after = "1"
    # Pad to 4 pairs (8 chars) – max_add_joint for thumbs is 4
    return f"{rotation}--{after}"


def _apply_pad_settings(settings, fc, prefix):
    """Override pad/bump settings on a FingerConfig from UI properties."""
    fc.pad_resolution_level = getattr(settings, f"{prefix}_pad_resolution")
    fc.pad_thickness_mm = getattr(settings, f"{prefix}_pad_thickness")
    max_int = getattr(settings, f"{prefix}_pad_max_intensity")
    n_kernels = getattr(settings, f"{prefix}_pad_kernel_count")
    max_joints = getattr(fc, "max_add_joint_slots", 5)

    fc.bump_type_list = []
    fc.bump_number_list = []
    fc.bump_max_height_intensity_mm_list = []
    fc.bump_height_intensity_list = []
    fc.bumps_spread_list = []
    fc.bumps_aspect_ratio_list = []
    fc.bump_rotation_deg_list = []
    fc.bump_center_angle_deg_list = []
    fc.bump_center_offset_list = []

    for _joint in range(max_joints):
        fc.bump_type_list.append('gaussian')
        fc.bump_number_list.append(n_kernels)
        fc.bump_max_height_intensity_mm_list.append(max_int)
        intensities, spreads, aspects = [], [], []
        rotations, offsets, angles = [], [], []
        for k in range(n_kernels):
            intensities.append(getattr(settings, f"{prefix}_pad_k{k}_intensity"))
            spreads.append(getattr(settings, f"{prefix}_pad_k{k}_spread"))
            aspects.append(getattr(settings, f"{prefix}_pad_k{k}_aspect_ratio"))
            rotations.append(getattr(settings, f"{prefix}_pad_k{k}_rotation"))
            offsets.append(getattr(settings, f"{prefix}_pad_k{k}_center_offset"))
            angles.append(getattr(settings, f"{prefix}_pad_k{k}_center_angle"))
        fc.bump_height_intensity_list.append(intensities)
        fc.bumps_spread_list.append(spreads)
        fc.bumps_aspect_ratio_list.append(aspects)
        fc.bump_rotation_deg_list.append(rotations)
        fc.bump_center_offset_list.append(offsets)
        fc.bump_center_angle_deg_list.append(angles)


def _randomize_ui_pad_from_config(settings, prefix, cfg, max_kernels=3):
    """Map sampled Config pad data into the add-on's per-digit pad UI."""
    counts = list(getattr(cfg, "bump_number_list", []))
    max_heights = list(getattr(cfg, "bump_max_height_intensity_mm_list", []))
    slot = 0
    for idx, count in enumerate(counts):
        height = max_heights[idx] if idx < len(max_heights) else 0.0
        if count > 0 and height > 1e-6:
            slot = idx
            break

    count = int(counts[slot]) if slot < len(counts) else 0
    count = max(0, min(count, max_kernels))
    max_height = float(max_heights[slot]) if slot < len(max_heights) else 0.0
    setattr(settings, f"{prefix}_show_pad", count > 0 and max_height > 1e-6)
    setattr(settings, f"{prefix}_pad_resolution", getattr(cfg, "pad_resolution_level", 2))
    setattr(settings, f"{prefix}_pad_thickness", getattr(cfg, "pad_thickness_mm", 1.0))
    setattr(settings, f"{prefix}_pad_max_intensity", max_height)
    setattr(settings, f"{prefix}_pad_kernel_count", count)

    def _slot_list(name):
        seq = getattr(cfg, name, [])
        return list(seq[slot]) if slot < len(seq) else []

    intensities = _slot_list("bump_height_intensity_list")
    spreads = _slot_list("bumps_spread_list")
    aspects = _slot_list("bumps_aspect_ratio_list")
    rotations = _slot_list("bump_rotation_deg_list")
    offsets = _slot_list("bump_center_offset_list")
    angles = _slot_list("bump_center_angle_deg_list")
    for k in range(max_kernels):
        setattr(settings, f"{prefix}_pad_k{k}_intensity", float(intensities[k]) if k < len(intensities) else 1.0)
        setattr(settings, f"{prefix}_pad_k{k}_spread", float(spreads[k]) if k < len(spreads) else 0.25)
        setattr(settings, f"{prefix}_pad_k{k}_aspect_ratio", float(aspects[k]) if k < len(aspects) else 1.0)
        setattr(settings, f"{prefix}_pad_k{k}_rotation", float(rotations[k]) if k < len(rotations) else 0.0)
        setattr(settings, f"{prefix}_pad_k{k}_center_offset", float(offsets[k]) if k < len(offsets) else 0.0)
        setattr(settings, f"{prefix}_pad_k{k}_center_angle", float(angles[k]) if k < len(angles) else 0.0)


def _set_finger_code_ui(settings, i, code):
    parts = str(code).split("-")
    while len(parts) < 3:
        parts.append("")
    rotation, before, after = parts[0], parts[1], parts[2]
    setattr(settings, f"finger_{i}_rotation", rotation if rotation in {"0", "1", "2", "3", "4"} else "0")
    setattr(settings, f"finger_{i}_before", "".join(c for c in before if c in "12"))
    setattr(settings, f"finger_{i}_after", "".join(c for c in after if c in "12") or "1")


def _set_thumb_code_ui(settings, i, code):
    parts = str(code).split("-")
    while len(parts) < 3:
        parts.append("")
    rotation, after = parts[0], parts[2]
    setattr(settings, f"thumb_{i}_rotation", rotation if rotation in {"0", "1"} else "0")
    setattr(settings, f"thumb_{i}_after", "".join(c for c in after if c in "12") or "1")


def randomize_parameters(context):
    """Sample ParamGen values into the add-on UI settings."""
    settings = context.scene.handgen_settings
    _set_generation_v2_dir(settings.generation_v2_dir)
    _ensure_matplotlib_mock()

    for mod_name in list(sys.modules.keys()):
        if mod_name in ("Config", "ParamGen", "PalmClass", "FingerClass", "HandClass", "utils"):
            del sys.modules[mod_name]

    from ParamGen import PalmParams, FingerParams

    old_auto_update = settings.auto_update
    settings.auto_update = False
    try:
        mode = settings.generation_mode
        palm_cfg = PalmParams(fix=False, thumb_side="random", generation_mode=mode).build(max_location_attempts=300)

        settings.palm_size_mm = palm_cfg.palm_size_mm
        settings.outline_sides = palm_cfg.outline_sides
        settings.outline_aspect_ratio = palm_cfg.outline_aspect_ratio
        settings.outline_longer_axis = palm_cfg.outline_longer_axis
        settings.outline_rotation_deg = palm_cfg.outline_rotation_deg
        settings.smoothing_iters = palm_cfg.smoothing_iters
        settings.smoothing_t = palm_cfg.smoothing_t
        settings.resolution_mm = palm_cfg.resolution_mm
        settings.palm_wall_thickness_mm = palm_cfg.palm_wall_thickness_mm
        settings.finger_number = min(max(int(palm_cfg.finger_number), 0), 8)
        settings.thumb_number = 0 if mode == 'finger_only' else min(max(int(palm_cfg.thumb_number), 1), 3)
        settings.thumb_side = palm_cfg.thumb_side if palm_cfg.thumb_side in {"left", "right", "random"} else "random"
        settings.use_custom_locations = True
        settings.copy_finger_settings = False
        settings.copy_thumb_settings = False

        # Keep these in robust, commonly used assembly ranges. ParamGen currently
        # owns palm/finger/thumb shape sampling, not these HandConfig values.
        settings.palm_extrude_height_mm = random.uniform(35.0, 55.0)
        settings.palm_thickness_mm = random.uniform(3.0, 6.0)
        settings.finger_palm_base_offset_mm = random.uniform(8.0, 14.0)

        settings.show_pad_settings = palm_cfg.bumps_number > 0 and palm_cfg.bump_max_height_intensity_mm > 1e-6
        settings.pad_resolution = palm_cfg.pad_resolution_level
        settings.pad_thickness_mm = palm_cfg.pad_thickness_mm
        settings.pad_max_intensity = palm_cfg.bump_max_height_intensity_mm
        settings.pad_kernel_count = min(int(palm_cfg.bumps_number), 5)
        for k in range(5):
            settings.__setattr__(f"pad_kernel_{k}_intensity", palm_cfg.bump_height_intensity_list[k] if k < len(palm_cfg.bump_height_intensity_list) else 1.0)
            settings.__setattr__(f"pad_kernel_{k}_spread", palm_cfg.bumps_spread_list[k] if k < len(palm_cfg.bumps_spread_list) else 0.25)
            settings.__setattr__(f"pad_kernel_{k}_aspect_ratio", palm_cfg.bumps_aspect_ratio_list[k] if k < len(palm_cfg.bumps_aspect_ratio_list) else 1.0)
            settings.__setattr__(f"pad_kernel_{k}_rotation", palm_cfg.bump_rotation_deg_list[k] if k < len(palm_cfg.bump_rotation_deg_list) else 0.0)
            settings.__setattr__(f"pad_kernel_{k}_center_offset", palm_cfg.bump_center_offset_list[k] if k < len(palm_cfg.bump_center_offset_list) else 0.0)
            settings.__setattr__(f"pad_kernel_{k}_center_angle", palm_cfg.bump_center_angle_deg_list[k] if k < len(palm_cfg.bump_center_angle_deg_list) else 0.0)

        for i in range(8):
            if i < settings.finger_number:
                fc = FingerParams(type="finger", fix=False).build(id=i)
                _set_finger_code_ui(settings, i, fc.code)
                setattr(settings, f"finger_{i}_location", palm_cfg.finger_location_list[i])
                setattr(settings, f"finger_{i}_angle", palm_cfg.finger_angle_deg_list[i])
                setattr(settings, f"finger_{i}_normal_offset", palm_cfg.finger_base_normal_offset_mm_list[i])
                setattr(settings, f"finger_{i}_side_offset", palm_cfg.finger_base_side_offset_mm_list[i])
                setattr(settings, f"finger_{i}_link_added_length", fc.link_added_length_mm_list[0] if fc.link_added_length_mm_list else 0.0)
                setattr(settings, f"finger_{i}_fingertip_scale", (
                    random.uniform(0.85, 1.15),
                    random.uniform(0.85, 1.15),
                    random.uniform(0.85, 1.15),
                ))
                _randomize_ui_pad_from_config(settings, f"finger_{i}", fc)
            else:
                setattr(settings, f"finger_{i}_show_pad", False)

        for i in range(3):
            if i < settings.thumb_number:
                tc = FingerParams(type="thumb", fix=False).build(id=i)
                _set_thumb_code_ui(settings, i, tc.code)
                setattr(settings, f"thumb_{i}_location", palm_cfg.thumb_location_list[i])
                setattr(settings, f"thumb_{i}_angle", palm_cfg.thumb_angle_deg_list[i])
                setattr(settings, f"thumb_{i}_normal_offset", palm_cfg.thumb_base_normal_offset_mm_list[i])
                setattr(settings, f"thumb_{i}_side_offset", palm_cfg.thumb_base_side_offset_mm_list[i])
                setattr(settings, f"thumb_{i}_link_added_length", tc.link_added_length_mm_list[0] if tc.link_added_length_mm_list else 0.0)
                setattr(settings, f"thumb_{i}_fingertip_scale", (
                    random.uniform(0.85, 1.15),
                    random.uniform(0.85, 1.15),
                    random.uniform(0.85, 1.15),
                ))
                _randomize_ui_pad_from_config(settings, f"thumb_{i}", tc)
            else:
                setattr(settings, f"thumb_{i}_show_pad", False)
    finally:
        settings.auto_update = old_auto_update


# ---------------------------------------------------------------------------
# Scene cleanup
# ---------------------------------------------------------------------------

def _remove_collection_recursive(col):
    for child_col in list(col.children):
        _remove_collection_recursive(child_col)
    for obj in list(col.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.collections.remove(col)


def _remove_object_tree(obj):
    for child in list(obj.children):
        _remove_object_tree(child)
    bpy.data.objects.remove(obj, do_unlink=True)


def _clear_generated_hand():
    """Remove all objects/collections from a previous generation."""
    # Clear joint sliders
    if hasattr(bpy.context.scene, 'handgen_joints'):
        bpy.context.scene.handgen_joints.clear()

    hand_col = bpy.data.collections.get("hand")
    if hand_col:
        _remove_collection_recursive(hand_col)

    palm_root = bpy.data.objects.get("PalmBody_root")
    if palm_root:
        _remove_object_tree(palm_root)

    to_remove = [
        obj for obj in bpy.data.objects
        if obj.name.lower().startswith(("viz_", "collider_", "palmbody",
                                        "screw_holes", "mount_holes"))
    ]
    for obj in to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    try:
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass


def _clear_template_objects():
    """Remove template objects that were appended from components.blend.

    Template objects live in a special collection called '_handgen_templates'.
    """
    tmpl_col = bpy.data.collections.get("_handgen_templates")
    if tmpl_col:
        _remove_collection_recursive(tmpl_col)
    try:
        bpy.ops.outliner.orphans_purge(
            do_local_ids=True, do_linked_ids=True, do_recursive=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Append helpers
# ---------------------------------------------------------------------------

def _append_all_from_blend(blend_path):
    """Append all objects and materials from a .blend file.

    Objects are placed into a hidden '_handgen_templates' collection so they
    don't clutter the user's scene but are available for the assembly code.
    Returns list of appended objects.
    """
    if not os.path.isfile(blend_path):
        print(f"[HandGenerator] components.blend not found: {blend_path}")
        return []

    # Create or reuse the templates collection
    tmpl_col = bpy.data.collections.get("_handgen_templates")
    if tmpl_col is None:
        tmpl_col = bpy.data.collections.new("_handgen_templates")
        bpy.context.scene.collection.children.link(tmpl_col)

    # Record existing objects before appending
    existing_objs = set(bpy.data.objects.keys())

    # Append all objects and materials
    with bpy.data.libraries.load(blend_path, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects
        data_to.materials = data_from.materials

    # Link appended objects to the templates collection
    appended = []
    for obj in data_to.objects:
        if obj is not None and obj.name not in existing_objs:
            try:
                tmpl_col.objects.link(obj)
            except RuntimeError:
                pass  # already linked
            appended.append(obj)

    # Also gather any child objects that were loaded implicitly
    new_objs = [o for o in bpy.data.objects if o.name not in existing_objs
                and o not in appended]
    for obj in new_objs:
        try:
            tmpl_col.objects.link(obj)
        except RuntimeError:
            pass
        appended.append(obj)

    # Evaluate the depsgraph so appended objects have correct matrix_world
    # (needed before excluding the collection, otherwise matrix_world is stale
    # and fingertip scaling in run_assembly reads wrong values)
    bpy.context.view_layer.update()

    # Hide the templates collection from viewport
    # Find the layer collection for _handgen_templates and exclude it
    def _find_layer_col(layer_col, name):
        if layer_col.name == name:
            return layer_col
        for child in layer_col.children:
            found = _find_layer_col(child, name)
            if found:
                return found
        return None

    vl = bpy.context.view_layer
    lc = _find_layer_col(vl.layer_collection, "_handgen_templates")
    if lc:
        lc.exclude = True

    return appended


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------

def _save_hand_configs(hand, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    hand.palm_cfg.save_config(os.path.join(output_dir, "palm_cfg.py"))
    for finger_cfg in hand.finger_cfgs:
        finger_cfg.save_config(os.path.join(output_dir, f"finger_cfg_{finger_cfg.id}.py"))
    for thumb_cfg in hand.thumb_cfgs:
        thumb_cfg.save_config(os.path.join(output_dir, f"thumb_cfg_{thumb_cfg.id}.py"))
    hand.hand_cfg.save_config(os.path.join(output_dir, "hand_cfg.py"))


def _apply_hand_visibility(context):
    """Apply group/name visibility filters to the currently generated hand."""
    settings = context.scene.handgen_settings

    finger_objs = set()
    thumb_objs = set()
    palm_objs = set()
    hand_col = bpy.data.collections.get("hand")
    if hand_col:
        for sub_col in hand_col.children:
            sub_name = sub_col.name.lower()
            if sub_name.startswith("finger_"):
                finger_objs.update(sub_col.objects)
            elif sub_name.startswith("thumb_"):
                thumb_objs.update(sub_col.objects)

    palm_root = bpy.data.objects.get("PalmBody_root")
    if palm_root:
        palm_objs.add(palm_root)
        stack = list(palm_root.children)
        while stack:
            c = stack.pop()
            palm_objs.add(c)
            stack.extend(c.children)

    for obj in bpy.data.objects:
        if obj.name.startswith("handgen_j"):
            continue

        visible = True
        if obj in finger_objs:
            visible = settings.show_fingers
        elif obj in thumb_objs:
            visible = settings.show_thumbs
        elif obj in palm_objs:
            visible = settings.show_palm
        else:
            continue

        name_lower = obj.name.lower()
        if visible and "collider" in name_lower:
            visible = settings.show_collision_mesh
        if visible and "revolute" in name_lower:
            visible = settings.show_revolute
        if visible and "viz" in name_lower:
            visible = settings.show_viz
        if visible and "attach" in name_lower:
            visible = settings.show_attach

        obj.hide_viewport = not visible
        obj.hide_render = not visible


def generate_hand(context):
    """Run the full hand generation pipeline inside Blender."""
    settings = context.scene.handgen_settings
    _set_generation_v2_dir(settings.generation_v2_dir)
    _ensure_matplotlib_mock()

    # Force-reload generation modules
    for mod_name in list(sys.modules.keys()):
        if mod_name in ("Config", "PalmClass", "FingerClass", "HandClass", "utils"):
            del sys.modules[mod_name]

    from Config import PalmConfig, FingerConfig, HandConfig
    from HandClass import Hand

    # --- Build PalmConfig ---
    palm_cfg = PalmConfig(data="fixed", detailed_viz=settings.detailed_viz)
    mode = settings.generation_mode
    thumb_number = 0 if mode == 'finger_only' else settings.thumb_number
    palm_cfg.generation_mode = mode
    palm_cfg.palm_size_mm = settings.palm_size_mm
    palm_cfg.palm_wall_thickness_mm = settings.palm_wall_thickness_mm
    palm_cfg.outline_sides = settings.outline_sides
    palm_cfg.outline_aspect_ratio = settings.outline_aspect_ratio
    palm_cfg.outline_longer_axis = settings.outline_longer_axis
    palm_cfg.outline_rotation_deg = settings.outline_rotation_deg
    palm_cfg.smoothing_iters = settings.smoothing_iters
    palm_cfg.smoothing_t = settings.smoothing_t
    palm_cfg.resolution_mm = settings.resolution_mm
    palm_cfg.finger_number = settings.finger_number
    palm_cfg.thumb_number = thumb_number
    palm_cfg.thumb_side = settings.thumb_side if thumb_number == 1 else 'random'

    # --- Pad kernel settings ---
    if settings.show_pad_settings:
        palm_cfg.pad_resolution_level = settings.pad_resolution
        palm_cfg.pad_thickness_mm = settings.pad_thickness_mm
        palm_cfg.bump_type = 'gaussian'
        palm_cfg.bump_max_height_intensity_mm = settings.pad_max_intensity
        palm_cfg.bumps_number = settings.pad_kernel_count

    # Reset lists that accumulate across calls
    palm_cfg.finger_angle_deg_list = []
    palm_cfg.thumb_angle_deg_list = []
    palm_cfg.finger_base_normal_offset_mm_list = []
    palm_cfg.thumb_base_normal_offset_mm_list = []
    palm_cfg.finger_base_side_offset_mm_list = []
    palm_cfg.thumb_base_side_offset_mm_list = []
    palm_cfg.finger_location_list = []
    palm_cfg.thumb_location_list = []
    palm_cfg.bump_height_intensity_list = []
    palm_cfg.bumps_spread_list = []
    palm_cfg.bumps_aspect_ratio_list = []
    palm_cfg.bump_rotation_deg_list = []
    palm_cfg.bump_center_angle_deg_list = []
    palm_cfg.bump_center_offset_list = []

    # Pre-populate per-finger/thumb lists so generate_values() won't override
    for i in range(settings.finger_number):
        fs = 0 if settings.copy_finger_settings else i
        palm_cfg.finger_angle_deg_list.append(
            getattr(settings, f"finger_{fs}_angle"))
        palm_cfg.finger_base_normal_offset_mm_list.append(
            getattr(settings, f"finger_{fs}_normal_offset"))
        palm_cfg.finger_base_side_offset_mm_list.append(
            getattr(settings, f"finger_{fs}_side_offset"))
        if settings.use_custom_locations:
            palm_cfg.finger_location_list.append(getattr(settings, f"finger_{i}_location"))
        else:
            palm_cfg.finger_location_list.append((i + 1) / (settings.finger_number + 1))
    for i in range(thumb_number):
        ts = 0 if settings.copy_thumb_settings else i
        palm_cfg.thumb_angle_deg_list.append(
            getattr(settings, f"thumb_{ts}_angle"))
        palm_cfg.thumb_base_normal_offset_mm_list.append(
            getattr(settings, f"thumb_{ts}_normal_offset"))
        palm_cfg.thumb_base_side_offset_mm_list.append(
            getattr(settings, f"thumb_{ts}_side_offset"))
        if settings.use_custom_locations:
            palm_cfg.thumb_location_list.append(getattr(settings, f"thumb_{i}_location"))
        else:
            palm_cfg.thumb_location_list.append((i + 1) / (thumb_number + 1))

    # Pre-populate per-kernel lists so generate_values() won't override
    if settings.show_pad_settings:
        for i in range(settings.pad_kernel_count):
            palm_cfg.bump_height_intensity_list.append(
                getattr(settings, f"pad_kernel_{i}_intensity"))
            palm_cfg.bumps_spread_list.append(
                getattr(settings, f"pad_kernel_{i}_spread"))
            palm_cfg.bumps_aspect_ratio_list.append(
                getattr(settings, f"pad_kernel_{i}_aspect_ratio"))
            palm_cfg.bump_rotation_deg_list.append(
                getattr(settings, f"pad_kernel_{i}_rotation"))
            palm_cfg.bump_center_offset_list.append(
                getattr(settings, f"pad_kernel_{i}_center_offset"))
            palm_cfg.bump_center_angle_deg_list.append(
                getattr(settings, f"pad_kernel_{i}_center_angle"))
    else:
        palm_cfg.bump_height_intensity_list = [1.0 for _ in range(palm_cfg.bumps_number)]
        palm_cfg.bumps_spread_list = [0.25 for _ in range(palm_cfg.bumps_number)]
        palm_cfg.bumps_aspect_ratio_list = [1.0 for _ in range(palm_cfg.bumps_number)]
        palm_cfg.bump_rotation_deg_list = [0.0 for _ in range(palm_cfg.bumps_number)]
        palm_cfg.bump_center_offset_list = [0.0 for _ in range(palm_cfg.bumps_number)]
        palm_cfg.bump_center_angle_deg_list = [0.0 for _ in range(palm_cfg.bumps_number)]

    palm_cfg.update()

    # --- Build FingerConfigs (code built from UI components) ---
    finger_cfgs = []
    for i in range(settings.finger_number):
        fs = 0 if settings.copy_finger_settings else i
        code = _build_finger_code(settings, fs)
        fc = FingerConfig(
            type='finger', data='fixed',
            code=code, id=i,
        )
        fc.fingertip_scale_factor = tuple(getattr(settings, f"finger_{fs}_fingertip_scale"))
        _link_len = getattr(settings, f"finger_{fs}_link_added_length")
        fc.link_added_length_mm_list = [_link_len] * len(fc.link_added_length_mm_list)
        if getattr(settings, f"finger_{i}_show_pad"):
            _apply_pad_settings(settings, fc, f"finger_{i}")
        finger_cfgs.append(fc)

    thumb_cfgs = []
    for i in range(thumb_number):
        ts = 0 if settings.copy_thumb_settings else i
        code = _build_thumb_code(settings, ts)
        tc = FingerConfig(
            type='thumb', data='fixed',
            code=code, id=i,
        )
        tc.fingertip_scale_factor = tuple(getattr(settings, f"thumb_{ts}_fingertip_scale"))
        _link_len = getattr(settings, f"thumb_{ts}_link_added_length")
        tc.link_added_length_mm_list = [_link_len] * len(tc.link_added_length_mm_list)
        if getattr(settings, f"thumb_{i}_show_pad"):
            _apply_pad_settings(settings, tc, f"thumb_{i}")
        thumb_cfgs.append(tc)

    # --- Build HandConfig ---
    hand_cfg = HandConfig()
    hand_cfg.palm_extrude_height_mm = settings.palm_extrude_height_mm
    hand_cfg.palm_thickness_mm = settings.palm_thickness_mm
    hand_cfg.palm_wall_thickness_mm = settings.palm_wall_thickness_mm
    hand_cfg.finger_palm_base_offset_mm = settings.finger_palm_base_offset_mm
    hand_cfg.collision_mesh = settings.generate_collision_mesh
    hand_cfg.thumb_mount_height_mm = (
        hand_cfg.palm_extrude_height_mm - hand_cfg.palm_thickness_mm
    )

    # --- Temp dir for npz ---
    tmp_dir = os.path.join(tempfile.gettempdir(), "handgen_blender")
    os.makedirs(tmp_dir, exist_ok=True)

    hand = Hand(
        palm_cfg=palm_cfg,
        finger_cfgs=finger_cfgs,
        thumb_cfgs=thumb_cfgs,
        hand_cfg=hand_cfg,
        root_dir=tmp_dir,
    )

    assembly_npz = hand.save_assembly_data()
    palm_npz = hand.save_palm_data()
    _save_hand_configs(hand, tmp_dir)

    if assembly_npz is None:
        print("[HandGenerator] Failed to generate assembly data.")
        return {'CANCELLED'}

    context.scene["handgen_last_output_dir"] = tmp_dir
    context.scene["handgen_last_assembly_npz"] = assembly_npz
    context.scene["handgen_last_palm_npz"] = palm_npz if palm_npz else ""

    # --- Clear previous hand and templates ---
    _clear_generated_hand()
    _clear_template_objects()

    # --- Load the blender_full_assembly module ---
    asm_mod_name = "blender_full_assembly"
    if asm_mod_name in sys.modules:
        del sys.modules[asm_mod_name]
    asm_path = os.path.join(GENERATION_DIR, "blender_full_assembly.py")
    spec = importlib.util.spec_from_file_location(asm_mod_name, asm_path)
    asm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asm_mod)

    # --- Append template objects from components.blend ---
    appended_objs = _append_all_from_blend(COMPONENTS_BLEND)

    # --- Build FingerAssembly without replacing the current Blender scene ---
    fa = asm_mod.FingerAssembly.from_existing_scene(assembly_npz, appended_objs)

    # --- Run assembly ---
    fa.run_assembly()

    # --- Palm generation ---
    if palm_npz and os.path.isfile(palm_npz):
        asm_mod.PalmMesh(palm_npz).generate()

    # --- Materials ---
    asm_mod.assign_viz_materials()

    # --- Origin offset ---
    if palm_npz and os.path.isfile(palm_npz):
        asm_mod.apply_origin_offset(palm_npz)

    # --- Clean up templates ---
    _clear_template_objects()

    # --- Set up joint controls ---
    _setup_joint_controls(context, asm_mod, assembly_npz)

    # --- Shade flat on all mesh objects ---
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.data:
            for poly in obj.data.polygons:
                poly.use_smooth = False

    # --- Visibility: group + name-based (AND logic) ---
    _apply_hand_visibility(context)

    # --- Enable cavity shading in all 3D viewports ---
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            shading = area.spaces[0].shading
            shading.show_cavity = True
            shading.cavity_type = 'BOTH'
            shading.cavity_ridge_factor = 2.5
            shading.cavity_valley_factor = 2.5
            shading.curvature_ridge_factor = 2.0
            shading.curvature_valley_factor = 2.0

    return {'FINISHED'}


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

class HANDGEN_OT_Generate(bpy.types.Operator):
    bl_idname = "handgen.generate"
    bl_label = "Generate Hand"
    bl_description = "Generate the robotic hand with current parameters"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            result = generate_hand(context)
            if result == {'FINISHED'}:
                self.report({'INFO'}, "Hand generated successfully")
            return result
        except Exception as e:
            self.report({'ERROR'}, f"Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class HANDGEN_OT_ClearHand(bpy.types.Operator):
    bl_idname = "handgen.clear"
    bl_label = "Clear Hand"
    bl_description = "Remove all generated hand objects"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        _clear_generated_hand()
        self.report({'INFO'}, "Hand cleared")
        return {'FINISHED'}


class HANDGEN_OT_RandomizeParameters(bpy.types.Operator):
    bl_idname = "handgen.randomize_parameters"
    bl_label = "Randomize Parameters"
    bl_description = "Randomize hand parameters using ParamGen, then generate the hand"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            randomize_parameters(context)
            result = generate_hand(context)
            if result == {'FINISHED'}:
                self.report({'INFO'}, "Parameters randomized and hand generated")
            return result
        except Exception as e:
            self.report({'ERROR'}, f"Randomized generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


class HANDGEN_OT_SaveHand(bpy.types.Operator):
    bl_idname = "handgen.save_hand"
    bl_label = "Save Hand"
    bl_description = "Generate and save configs, NPZ data, Blend file, URDF, and URDF meshes"
    bl_options = {'REGISTER'}

    def execute(self, context):
        try:
            settings = context.scene.handgen_settings
            base_dir = bpy.path.abspath(settings.save_output_dir)
            hand_name = "".join(c if c not in '<>:"/\\|?*' else "_" for c in settings.save_hand_name).strip()
            if not hand_name:
                hand_name = "hand_addon"
            output_dir = os.path.abspath(os.path.join(base_dir, hand_name))
            os.makedirs(output_dir, exist_ok=True)

            result = generate_hand(context)
            if result != {'FINISHED'}:
                return result

            tmp_dir = context.scene.get("handgen_last_output_dir", "")
            assembly_npz = context.scene.get("handgen_last_assembly_npz", "")
            if not tmp_dir or not os.path.isdir(tmp_dir) or not assembly_npz:
                self.report({'ERROR'}, "No generated hand data found to save")
                return {'CANCELLED'}

            for filename in os.listdir(tmp_dir):
                src = os.path.join(tmp_dir, filename)
                dst = os.path.join(output_dir, filename)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)

            saved_assembly_npz = os.path.join(output_dir, "assembly_data.npz")

            blend_path = os.path.join(output_dir, "assembled_hand_model.blend")
            bpy.ops.wm.save_as_mainfile(
                filepath=blend_path,
                compress=False,
                copy=True,
                relative_remap=False,
            )

            asm_mod_name = "blender_full_assembly"
            if asm_mod_name in sys.modules:
                asm_mod = sys.modules[asm_mod_name]
            else:
                asm_path = os.path.join(GENERATION_DIR, "blender_full_assembly.py")
                spec = importlib.util.spec_from_file_location(asm_mod_name, asm_path)
                asm_mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(asm_mod)

            urdf_dir = os.path.join(output_dir, "urdf_hand_export")
            asm_mod.ExportURDF(
                GENERATION_DIR,
                assembly_npz_path=saved_assembly_npz,
                export_dir=urdf_dir,
            ).run()

            self.report({'INFO'}, f"Saved hand to {output_dir}")
            return {'FINISHED'}
        except Exception as e:
            self.report({'ERROR'}, f"Save failed: {e}")
            import traceback
            traceback.print_exc()
            return {'CANCELLED'}


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

class HANDGEN_PT_MainPanel(bpy.types.Panel):
    bl_label = "Hand Generator V2"
    bl_idname = "HANDGEN_PT_main"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hand Gen V2"

    def draw(self, context):
        layout = self.layout
        s = context.scene.handgen_settings

        row = layout.row(align=True)
        row.scale_y = 1.5
        row.operator("handgen.generate", icon='MOD_BUILD')
        row.operator("handgen.clear", icon='TRASH')
        layout.operator("handgen.randomize_parameters", icon='FILE_REFRESH')
        layout.separator()
        layout.prop(s, "generation_mode")
        layout.separator()
        layout.prop(s, "generation_v2_dir")
        if not _is_valid_generation_v2_dir(s.generation_v2_dir):
            layout.label(text="Invalid generation_v2 folder", icon='ERROR')
        layout.separator()
        layout.prop(s, "save_output_dir")
        layout.prop(s, "save_hand_name")
        layout.operator("handgen.save_hand", icon='FILE_TICK')
        layout.separator()
        layout.prop(s, "auto_update")


class HANDGEN_PT_PalmShape(bpy.types.Panel):
    bl_label = "Palm Shape"
    bl_idname = "HANDGEN_PT_palm_shape"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hand Gen V2"
    bl_parent_id = "HANDGEN_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = context.scene.handgen_settings
        layout.prop(s, "palm_size_mm")
        layout.prop(s, "outline_sides")
        layout.prop(s, "outline_aspect_ratio")
        layout.prop(s, "outline_longer_axis")
        layout.prop(s, "outline_rotation_deg")
        layout.prop(s, "resolution_mm")
        layout.prop(s, "smoothing_iters")
        layout.prop(s, "smoothing_t")
        layout.separator()
        layout.prop(s, "show_pad_settings")
        if s.show_pad_settings:
            pad_box = layout.box()
            pad_box.label(text="Palm Pad Bumps")
            pad_box.prop(s, "pad_resolution")
            pad_box.prop(s, "pad_thickness_mm")
            pad_box.prop(s, "pad_max_intensity")
            pad_box.prop(s, "pad_kernel_count")
            for i in range(s.pad_kernel_count):
                kbox = pad_box.box()
                kbox.label(text=f"Kernel {i + 1}")
                row = kbox.row(align=True)
                row.prop(s, f"pad_kernel_{i}_intensity")
                row.prop(s, f"pad_kernel_{i}_spread")
                row = kbox.row(align=True)
                row.prop(s, f"pad_kernel_{i}_aspect_ratio")
                row.prop(s, f"pad_kernel_{i}_rotation")
                row = kbox.row(align=True)
                row.prop(s, f"pad_kernel_{i}_center_offset")
                row.prop(s, f"pad_kernel_{i}_center_angle")


class HANDGEN_PT_FingersAndThumbs(bpy.types.Panel):
    bl_label = "Fingers & Thumbs"
    bl_idname = "HANDGEN_PT_fingers"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hand Gen V2"
    bl_parent_id = "HANDGEN_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = context.scene.handgen_settings
        finger_only = s.generation_mode == 'finger_only'
        layout.prop(s, "finger_number")
        if not finger_only:
            layout.prop(s, "thumb_number")
            if s.thumb_number == 1:
                layout.prop(s, "thumb_side")
            else:
                layout.label(text="Thumb side: automatic for multiple thumbs")
        layout.prop(s, "use_custom_locations")
        layout.separator()

        # Per-finger panels
        for i in range(s.finger_number):
            geometry_copied = i > 0 and s.copy_finger_settings
            box = layout.box()
            box.label(text=f"Finger {i + 1}")
            if i == 0:
                box.prop(s, "copy_finger_settings")
            elif geometry_copied:
                box.label(text="Geometry copied from Finger 1; pad and location stay per-finger")
            if s.use_custom_locations:
                box.prop(s, f"finger_{i}_location")
            row = box.row(align=True)
            row.enabled = not geometry_copied
            row.prop(s, f"finger_{i}_rotation", text="Rot")
            row.prop(s, f"finger_{i}_before", text="Before")
            row.prop(s, f"finger_{i}_after", text="After")
            row = box.row(align=True)
            row.enabled = not geometry_copied
            row.prop(s, f"finger_{i}_angle")
            row.prop(s, f"finger_{i}_normal_offset")
            row.prop(s, f"finger_{i}_side_offset")
            row = box.row(align=True)
            row.enabled = not geometry_copied
            row.prop(s, f"finger_{i}_fingertip_scale")
            row = box.row()
            row.enabled = not geometry_copied
            row.prop(s, f"finger_{i}_link_added_length")
            box.prop(s, f"finger_{i}_show_pad")
            if getattr(s, f"finger_{i}_show_pad"):
                pbox = box.box()
                pbox.prop(s, f"finger_{i}_pad_resolution")
                pbox.prop(s, f"finger_{i}_pad_thickness")
                pbox.prop(s, f"finger_{i}_pad_max_intensity")
                pbox.prop(s, f"finger_{i}_pad_kernel_count")
                for j in range(getattr(s, f"finger_{i}_pad_kernel_count")):
                    kbox = pbox.box()
                    kbox.label(text=f"Kernel {j + 1}")
                    row = kbox.row(align=True)
                    row.prop(s, f"finger_{i}_pad_k{j}_intensity")
                    row.prop(s, f"finger_{i}_pad_k{j}_spread")
                    row = kbox.row(align=True)
                    row.prop(s, f"finger_{i}_pad_k{j}_aspect_ratio")
                    row.prop(s, f"finger_{i}_pad_k{j}_rotation")
                    row = kbox.row(align=True)
                    row.prop(s, f"finger_{i}_pad_k{j}_center_offset")
                    row.prop(s, f"finger_{i}_pad_k{j}_center_angle")

        layout.separator()

        if not finger_only:
            # Per-thumb panels
            for i in range(s.thumb_number):
                geometry_copied = i > 0 and s.copy_thumb_settings
                box = layout.box()
                box.label(text=f"Thumb {i + 1}")
                if i == 0:
                    box.prop(s, "copy_thumb_settings")
                elif geometry_copied:
                    box.label(text="Geometry copied from Thumb 1; pad and location stay per-thumb")
                if s.use_custom_locations:
                    box.prop(s, f"thumb_{i}_location")
                row = box.row(align=True)
                row.enabled = not geometry_copied
                row.prop(s, f"thumb_{i}_rotation", text="Rot")
                row.prop(s, f"thumb_{i}_after", text="Joints")
                row = box.row(align=True)
                row.enabled = not geometry_copied
                row.prop(s, f"thumb_{i}_angle")
                row.prop(s, f"thumb_{i}_normal_offset")
                row.prop(s, f"thumb_{i}_side_offset")
                row = box.row(align=True)
                row.enabled = not geometry_copied
                row.prop(s, f"thumb_{i}_fingertip_scale")
                row = box.row()
                row.enabled = not geometry_copied
                row.prop(s, f"thumb_{i}_link_added_length")
                box.prop(s, f"thumb_{i}_show_pad")
                if getattr(s, f"thumb_{i}_show_pad"):
                    pbox = box.box()
                    pbox.prop(s, f"thumb_{i}_pad_resolution")
                    pbox.prop(s, f"thumb_{i}_pad_thickness")
                    pbox.prop(s, f"thumb_{i}_pad_max_intensity")
                    pbox.prop(s, f"thumb_{i}_pad_kernel_count")
                    for j in range(getattr(s, f"thumb_{i}_pad_kernel_count")):
                        kbox = pbox.box()
                        kbox.label(text=f"Kernel {j + 1}")
                        row = kbox.row(align=True)
                        row.prop(s, f"thumb_{i}_pad_k{j}_intensity")
                        row.prop(s, f"thumb_{i}_pad_k{j}_spread")
                        row = kbox.row(align=True)
                        row.prop(s, f"thumb_{i}_pad_k{j}_aspect_ratio")
                        row.prop(s, f"thumb_{i}_pad_k{j}_rotation")
                        row = kbox.row(align=True)
                        row.prop(s, f"thumb_{i}_pad_k{j}_center_offset")
                        row.prop(s, f"thumb_{i}_pad_k{j}_center_angle")


class HANDGEN_PT_HandGeometry(bpy.types.Panel):
    bl_label = "Hand Geometry"
    bl_idname = "HANDGEN_PT_hand_geom"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hand Gen V2"
    bl_parent_id = "HANDGEN_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = context.scene.handgen_settings
        layout.prop(s, "palm_extrude_height_mm")
        layout.prop(s, "palm_thickness_mm")
        layout.prop(s, "palm_wall_thickness_mm")
        layout.prop(s, "finger_palm_base_offset_mm")


class HANDGEN_PT_Options(bpy.types.Panel):
    bl_label = "Options"
    bl_idname = "HANDGEN_PT_options"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hand Gen V2"
    bl_parent_id = "HANDGEN_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        s = context.scene.handgen_settings
        layout.prop(s, "detailed_viz")
        layout.prop(s, "generate_collision_mesh")
        layout.separator()
        layout.prop(s, "show_collision_mesh")
        layout.prop(s, "show_revolute")
        layout.prop(s, "show_viz")
        layout.prop(s, "show_attach")
        layout.prop(s, "show_fingers")
        layout.prop(s, "show_thumbs")
        layout.prop(s, "show_palm")
        layout.separator()
        layout.prop(s, "show_overlays")


# ---------------------------------------------------------------------------
# Joint control
# ---------------------------------------------------------------------------

def _handgen_joint_value_update(self, context):
    from mathutils import Matrix, Vector
    v = max(self.min_val, min(self.max_val, self.value))
    if abs(v - self.value) > 1e-9:
        self.value = v
        return
    frame_obj = bpy.data.objects.get(self.link_frame_name)
    if frame_obj is None:
        return
    axis = Vector((self.axis_x, self.axis_y, self.axis_z))
    if axis.length < 1e-9:
        axis = Vector((0, 0, 1))
    else:
        axis.normalize()
    frame_obj.matrix_local = Matrix.Rotation(v, 4, axis)


class HANDGEN_JointItem(bpy.types.PropertyGroup):
    name: bpy.props.StringProperty()
    value: bpy.props.FloatProperty(
        name="Angle", default=0.0,
        min=-3.14159, max=3.14159,
        update=_handgen_joint_value_update,
    )
    min_val: bpy.props.FloatProperty(default=-3.14159)
    max_val: bpy.props.FloatProperty(default=3.14159)
    axis_x: bpy.props.FloatProperty(default=0.0)
    axis_y: bpy.props.FloatProperty(default=0.0)
    axis_z: bpy.props.FloatProperty(default=1.0)
    link_frame_name: bpy.props.StringProperty()
    chain_id: bpy.props.StringProperty()


def _setup_joint_controls(context, asm_mod, assembly_npz):
    """Build joint control empties using the same chain logic as ExportURDF."""
    import re, math
    from mathutils import Matrix, Vector

    context.scene.handgen_joints.clear()

    old = [o for o in bpy.data.objects if o.name.startswith("handgen_j")]
    for o in old:
        bpy.data.objects.remove(o, do_unlink=True)

    hand_col = bpy.data.collections.get("hand")
    palm_root = bpy.data.objects.get("PalmBody_root")
    if not hand_col or not palm_root:
        return

    bpy.context.view_layer.update()

    # Instantiate ExportURDF for its helper methods (no actual export)
    exp = object.__new__(asm_mod.ExportURDF)
    exp.collection_name = "hand"
    exp.attach_thresh = 0.0005
    exp.Matrix = Matrix
    exp._mesh_rel_cache = {}
    exp.default_joint_limit = {"effort": 0.95, "velocity": 8.48, "lower": -0.47, "upper": 2.443}
    exp.joint_limit_by_instance = {}
    exp.joint_id_by_instance = {}
    exp.expected_joint_count_by_subgroup = {}

    # Load joint limits from npz
    if assembly_npz and os.path.isfile(assembly_npz):
        try:
            ad = np.load(assembly_npz, allow_pickle=True)
            ec = np.array(ad.get("elem_counter", []), dtype=int)
            et = np.array(ad.get("elem_types", []), dtype=object)
            gr = np.array(ad.get("groups", []), dtype=object)
            jlo = np.array(ad.get("joint_lower", []), dtype=float)
            jhi = np.array(ad.get("joint_upper", []), dtype=float)
            jid = np.array(ad.get("joint_id", []), dtype=float)
            n = int(min(len(ec), len(et), len(gr), len(jlo), len(jhi), len(jid)))
            for i in range(n):
                if str(et[i]) != "joint":
                    continue
                gi = str(gr[i])[0].lower()
                if gi not in ("f", "t"):
                    continue
                try:
                    if math.isfinite(float(jid[i])):
                        exp.joint_id_by_instance[(gi, int(ec[i]))] = int(float(jid[i]))
                except Exception:
                    pass
                lo, hi = float(jlo[i]), float(jhi[i])
                if math.isfinite(lo) and math.isfinite(hi):
                    exp.joint_limit_by_instance[(gi, int(ec[i]))] = {
                        "lower": lo, "upper": hi}
        except Exception:
            pass

    # --- Helpers (mirrors ExportURDF.build_urdf internals) ---
    def first_order_in_coll(coll):
        def _collect(c):
            out = list(c.objects)
            for ch in c.children:
                out.extend(_collect(ch))
            return out
        objs = _collect(coll)
        imm = [o for o in objs if o.parent is None or o.parent not in objs]
        imm.sort(key=lambda o: o.name.lower())
        return imm

    def all_attach_in_coll(coll):
        attaches = []
        for o in first_order_in_coll(coll):
            attaches.extend(exp.children_with_key(o, "attach_", mesh_only=True))
        return attaches

    def min_attach_dist(pa_obj, coll):
        att = all_attach_in_coll(coll)
        if not att:
            return float("inf")
        pa = exp.world_loc(pa_obj)
        return min((pa - exp.world_loc(a)).length for a in att)

    def remaining_attaches(mo, used):
        return [a for a in exp.children_with_key(mo, "attach_", mesh_only=True) if a not in used]

    def _joint_limit(rev_obj):
        if rev_obj is None:
            return None
        try:
            name = re.sub(r"\.\d+$", "", rev_obj.name)
            m = re.search(r"_([ft])(\d+)$", name.lower())
            if not m:
                return None
            return exp.joint_limit_by_instance.get((m.group(1), int(m.group(2))))
        except Exception:
            return None

    def _joint_id(rev_obj):
        if rev_obj is None:
            return 0
        try:
            name = re.sub(r"\.\d+$", "", rev_obj.name)
            m = re.search(r"_([ft])(\d+)$", name.lower())
            if not m:
                return 0
            return exp.joint_id_by_instance.get((m.group(1), int(m.group(2))), 0)
        except Exception:
            return 0

    # --- Match palm attaches to subcollections ---
    palm_attaches = [a for a in exp.children_with_key(palm_root, "attach_", mesh_only=True)
                     if a.name.lower().startswith("attach_finger_")
                     or a.name.lower().startswith("attach_thumb_")]

    subcolls = list(hand_col.children)
    used_subcolls = set()

    for pa in sorted(palm_attaches, key=lambda o: o.name.lower()):
        best_sub, best_d = None, float("inf")
        for sub in subcolls:
            if sub.name in used_subcolls:
                continue
            d = min_attach_dist(pa, sub)
            if d < best_d:
                best_sub, best_d = sub, d
        if best_sub is None or best_d > exp.attach_thresh:
            continue
        used_subcolls.add(best_sub.name)

        # --- Traverse the chain (same logic as export_chain_for_collection) ---
        chain_coll = best_sub
        chain_name = best_sub.name
        mains = [o for o in first_order_in_coll(chain_coll) if exp.type_of_main(o)]
        if not mains:
            continue

        # Chain token for naming (f1, f2, t1, ...)
        chain_tok = "f0"
        try:
            s = chain_name.lower()
            if s.startswith("finger_"):
                chain_tok = f"f{int(s.split('_')[-1])}"
            elif s.startswith("thumb_"):
                chain_tok = f"t{int(s.split('_')[-1])}"
        except Exception:
            pass

        visited = set()
        last_main = pa
        last_main_mainobj = None
        last_attaches = [pa]
        used_attaches = {pa}
        pending_rev = None

        # Collect: list of (revolute_obj, list_of_main_roots_in_child_link)
        # Each entry = one joint. The child link's main roots should be re-parented.
        urdf_links = [[]]  # urdf_links[0] = mains in first (parent) link
        joint_revs = []     # revolute objects, one per joint
        joint_extra_children = []  # extra objects to re-parent under jfr (w* viz splits)

        while True:
            candidates = [mo for mo in mains if mo not in visited]
            if not candidates:
                break

            best = (None, None, None, float("inf"))
            for mo in candidates:
                att = exp.children_with_key(mo, "attach_", mesh_only=True)
                if not att:
                    continue
                a, b, d = exp.nearest_attach_pair(last_attaches, att)
                if a and b and d < best[3]:
                    best = (mo, a, b, d)

            # Fallback: try all attaches on last main object
            if best[0] is None or best[3] > exp.attach_thresh:
                fb_obj = last_main_mainobj if last_main_mainobj else last_main
                prev_all = exp.children_with_key(fb_obj, "attach_", mesh_only=True) if fb_obj else []
                if prev_all:
                    best2 = (None, None, None, float("inf"))
                    for mo in candidates:
                        att = exp.children_with_key(mo, "attach_", mesh_only=True)
                        if not att:
                            continue
                        a2, b2, d2 = exp.nearest_attach_pair(prev_all, att)
                        if a2 and b2 and d2 < best2[3]:
                            best2 = (mo, a2, b2, d2)
                    if best2[0] is not None and best2[3] <= exp.attach_thresh:
                        best = best2

            if best[0] is None or best[3] > exp.attach_thresh:
                break

            mo, a_prev, b_this, dist = best
            visited.add(mo)
            if a_prev:
                used_attaches.add(a_prev)
            used_attaches.add(b_this)
            kind = exp.type_of_main(mo)

            # Handle deferred revolute (creates new URDF link)
            if pending_rev is not None:
                joint_revs.append(pending_rev)
                joint_extra_children.append([])
                urdf_links.append([])  # start new child link
                pending_rev = None

            if kind == "link":
                urdf_links[-1].append(mo)
            elif kind == "n":
                revs = exp.children_with_key(mo, "revolute", mesh_only=True)
                urdf_links[-1].append(mo)
                if revs:
                    pending_rev = revs[0]
            elif kind == "w":
                revs = exp.children_with_key(mo, "revolute", mesh_only=True)
                rev = revs[0] if revs else None
                vp_this = b_this.parent if b_this else None
                if vp_this is not None:
                    parent_of_viz = vp_this.parent
                    all_viz = exp.children_with_key(mo, "viz_", mesh_only=False)
                    other_viz = [v for v in all_viz if v.parent != parent_of_viz]
                    urdf_links[-1].append(mo)
                    if rev is not None and other_viz:
                        # w* immediate joint: sibling_viz stays, other_viz to child
                        joint_revs.append(rev)
                        joint_extra_children.append(other_viz)
                        urdf_links.append([])
                    elif rev is not None:
                        pending_rev = rev
                else:
                    # No attach parent: entire w* goes to child link
                    if rev is not None:
                        joint_revs.append(rev)
                        joint_extra_children.append([])
                        urdf_links.append([mo])
                    else:
                        urdf_links[-1].append(mo)

            last_main = mo
            last_main_mainobj = mo
            last_attaches = remaining_attaches(mo, used_attaches)

        # Handle final pending revolute
        if pending_rev is not None:
            joint_revs.append(pending_rev)
            joint_extra_children.append([])
            urdf_links.append([])

        if not joint_revs:
            continue

        # --- Create joint empties and re-parent ---
        for j_idx, rev_obj in enumerate(joint_revs):
            # Parent link: first main root of the parent URDF link
            parent_mains = urdf_links[j_idx]
            if not parent_mains:
                continue
            parent_root = parent_mains[0]

            # Child link: all mains in the child URDF link
            child_mains = urdf_links[j_idx + 1] if j_idx + 1 < len(urdf_links) else []

            local_axis = exp.joint_axis_in_joint_frame_local(rev_obj)
            lim = _joint_limit(rev_obj) or exp.default_joint_limit
            k = _joint_id(rev_obj)

            # Joint offset empty
            jof_name = f"handgen_jof_{chain_name}_j{j_idx}"
            jof = bpy.data.objects.new(jof_name, None)
            jof.empty_display_size = 0.001
            chain_coll.objects.link(jof)
            jof.parent = parent_root
            jof.matrix_parent_inverse = Matrix.Identity(4)
            jof.matrix_basis = parent_root.matrix_world.inverted() @ rev_obj.matrix_world

            # Joint frame empty
            jfr_name = f"handgen_jfr_{chain_name}_j{j_idx}"
            jfr = bpy.data.objects.new(jfr_name, None)
            jfr.empty_display_size = 0.001
            chain_coll.objects.link(jfr)
            jfr.parent = jof
            jfr.matrix_parent_inverse = Matrix.Identity(4)
            jfr.matrix_basis = Matrix.Identity(4)

            bpy.context.view_layer.update()

            # Re-parent child link's main roots under jfr
            for child_mo in child_mains:
                wm = child_mo.matrix_world.copy()
                child_mo.parent = jfr
                child_mo.matrix_parent_inverse = jfr.matrix_world.inverted() @ wm @ child_mo.matrix_basis.inverted()

            # Re-parent extra children from w* viz splits under jfr
            extras = joint_extra_children[j_idx] if j_idx < len(joint_extra_children) else []
            for extra_obj in extras:
                wm = extra_obj.matrix_world.copy()
                extra_obj.parent = jfr
                extra_obj.matrix_parent_inverse = jfr.matrix_world.inverted() @ wm @ extra_obj.matrix_basis.inverted()

            bpy.context.view_layer.update()

            # Register joint slider
            joint_name = f"J_{chain_tok}_{j_idx + 1}_{k}"
            item = context.scene.handgen_joints.add()
            item.name = joint_name
            item.axis_x = local_axis.x
            item.axis_y = local_axis.y
            item.axis_z = local_axis.z
            item.link_frame_name = jfr_name
            item.chain_id = chain_name
            item.min_val = float(lim.get("lower", -3.14159))
            item.max_val = float(lim.get("upper", 3.14159))


class HANDGEN_OT_ResetJoints(bpy.types.Operator):
    bl_idname = "handgen.reset_joints"
    bl_label = "Reset All Joints"
    bl_description = "Reset all joint angles to zero"

    def execute(self, context):
        for j in context.scene.handgen_joints:
            j.value = 0.0
        return {'FINISHED'}


def _pose_dir():
    blend = bpy.data.filepath
    if blend:
        return os.path.dirname(blend)
    return os.path.join(tempfile.gettempdir(), "handgen_blender")


def _recorded_pose_path():
    return os.path.join(_pose_dir(), "recorded_pose.json")


def _recorded_initial_pose_path():
    return os.path.join(_pose_dir(), "recorded_initial_pose.json")


class HANDGEN_OT_RecordPose(bpy.types.Operator):
    bl_idname = "handgen.record_pose"
    bl_label = "Record Pose"
    bl_description = "Record current joint values as target pose"

    def execute(self, context):
        import json
        joints = context.scene.handgen_joints
        if not joints:
            self.report({'WARNING'}, "No joints to record")
            return {'CANCELLED'}
        pose = {j.name: j.value for j in joints}
        path = _recorded_pose_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(pose, f, indent=2)
        self.report({'INFO'}, f"Target pose recorded ({len(pose)} joints)")
        return {'FINISHED'}


class HANDGEN_OT_RecordInitialPose(bpy.types.Operator):
    bl_idname = "handgen.record_initial_pose"
    bl_label = "Record Initial Pose"
    bl_description = "Record current joint values as initial pose for replay"

    def execute(self, context):
        import json
        joints = context.scene.handgen_joints
        if not joints:
            self.report({'WARNING'}, "No joints to record")
            return {'CANCELLED'}
        pose = {j.name: j.value for j in joints}
        path = _recorded_initial_pose_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(pose, f, indent=2)
        self.report({'INFO'}, f"Initial pose recorded ({len(pose)} joints)")
        return {'FINISHED'}


class HANDGEN_OT_PlayReplay(bpy.types.Operator):
    bl_idname = "handgen.play_replay"
    bl_label = "Play"
    bl_description = "Interpolate from initial pose to recorded pose"

    _timer = None
    _step = 0
    _total_steps = 1
    _targets = {}
    _starts = {}

    def execute(self, context):
        import json
        path = _recorded_pose_path()
        if not os.path.isfile(path):
            self.report({'WARNING'}, "No recorded pose found")
            return {'CANCELLED'}
        with open(path, 'r') as f:
            self._targets = json.load(f)
        # Load initial pose if available, otherwise default to zeros
        init_path = _recorded_initial_pose_path()
        if os.path.isfile(init_path):
            with open(init_path, 'r') as f:
                self._starts = json.load(f)
        else:
            self._starts = {}
        self._total_steps = max(1, context.scene.handgen_replay_steps)
        self._step = 0
        # Set joints to initial pose
        for j in context.scene.handgen_joints:
            j.value = self._starts.get(j.name, 0.0)
        duration = max(0.1, context.scene.handgen_replay_duration)
        interval = duration / self._total_steps
        self._timer = context.window_manager.event_timer_add(
            interval, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        if event.type == 'TIMER':
            self._step += 1
            t = self._step / self._total_steps
            if t >= 1.0:
                t = 1.0
            for j in context.scene.handgen_joints:
                start = self._starts.get(j.name, 0.0)
                target = self._targets.get(j.name, 0.0)
                j.value = start + (target - start) * t
            context.area.tag_redraw()
            if self._step >= self._total_steps:
                context.window_manager.event_timer_remove(self._timer)
                self._timer = None
                return {'FINISHED'}
        elif event.type == 'ESC':
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
            self.report({'INFO'}, "Replay cancelled")
            return {'CANCELLED'}
        return {'PASS_THROUGH'}


class HANDGEN_OT_AddToTimeline(bpy.types.Operator):
    bl_idname = "handgen.add_to_timeline"
    bl_label = "Add to Timeline"
    bl_description = "Keyframe the replay motion into the timeline (replaces existing)"

    def execute(self, context):
        import json
        from mathutils import Matrix, Vector

        path = _recorded_pose_path()
        if not os.path.isfile(path):
            self.report({'WARNING'}, "No recorded target pose found")
            return {'CANCELLED'}
        with open(path, 'r') as f:
            targets = json.load(f)

        init_path = _recorded_initial_pose_path()
        starts = {}
        if os.path.isfile(init_path):
            with open(init_path, 'r') as f:
                starts = json.load(f)

        joints = context.scene.handgen_joints
        if not joints:
            self.report({'WARNING'}, "No joints")
            return {'CANCELLED'}

        total_steps = max(1, context.scene.handgen_replay_steps)
        duration = max(0.1, context.scene.handgen_replay_duration)
        fps = context.scene.render.fps
        total_frames = max(total_steps, int(round(duration * fps)))

        # Collect jfr objects and clear old keyframes
        jfr_objs = {}
        for j in joints:
            obj = bpy.data.objects.get(j.link_frame_name)
            if obj is None:
                continue
            jfr_objs[j.name] = obj
            obj.rotation_mode = 'QUATERNION'
            if obj.animation_data and obj.animation_data.action:
                bpy.data.actions.remove(obj.animation_data.action)

        # Insert keyframes at evenly spaced frames
        for step in range(total_steps + 1):
            frame = 1 + int(round(step * total_frames / total_steps))
            t = step / total_steps
            for j in joints:
                obj = jfr_objs.get(j.name)
                if obj is None:
                    continue
                start = starts.get(j.name, 0.0)
                target = targets.get(j.name, 0.0)
                angle = start + (target - start) * t
                axis = Vector((j.axis_x, j.axis_y, j.axis_z))
                if axis.length < 1e-9:
                    axis = Vector((0, 0, 1))
                else:
                    axis.normalize()
                quat = Matrix.Rotation(angle, 4, axis).to_quaternion()
                obj.rotation_quaternion = quat
                obj.keyframe_insert(data_path='rotation_quaternion', frame=frame)

        # Set scene frame range
        context.scene.frame_start = 1
        context.scene.frame_end = 1 + total_frames
        context.scene.frame_current = 1

        self.report({'INFO'}, f"Keyframed {len(jfr_objs)} joints over {total_frames} frames ({duration}s)")
        return {'FINISHED'}


class HANDGEN_PT_JointControl(bpy.types.Panel):
    bl_label = "Joint Control"
    bl_idname = "HANDGEN_PT_joints"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hand Gen V2"
    bl_parent_id = "HANDGEN_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        joints = context.scene.handgen_joints

        if not joints:
            layout.label(text="No joints (generate hand first)")
            return

        row = layout.row(align=True)
        row.operator("handgen.reset_joints")
        row.operator("handgen.record_initial_pose", icon='SNAP_FACE', text="Initial")
        row.operator("handgen.record_pose", icon='REC', text="Target")

        box = layout.box()
        box.label(text="Replay")
        box.prop(context.scene, "handgen_replay_steps")
        box.prop(context.scene, "handgen_replay_duration")
        row = box.row(align=True)
        row.operator("handgen.play_replay", icon='PLAY')
        row.operator("handgen.add_to_timeline", icon='ACTION')

        layout.separator()

        chains = {}
        for j in joints:
            chains.setdefault(j.chain_id, []).append(j)

        for chain_id in sorted(chains.keys()):
            box = layout.box()
            box.label(text=chain_id.replace('_', ' ').title())
            for j in chains[chain_id]:
                box.prop(j, "value", slider=True, text=j.name)


# ---------------------------------------------------------------------------
# Auto-update timer
# ---------------------------------------------------------------------------

_auto_update_timer = None


def _auto_update_cb():
    global _auto_update_timer
    _auto_update_timer = None
    try:
        if bpy.context.scene and hasattr(bpy.context.scene, 'handgen_settings'):
            if bpy.context.scene.handgen_settings.auto_update:
                bpy.ops.handgen.generate()
    except Exception:
        import traceback
        traceback.print_exc()
    return None


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    HANDGEN_JointItem,
    HANDGEN_PG_Settings,
    HANDGEN_OT_Generate,
    HANDGEN_OT_ClearHand,
    HANDGEN_OT_RandomizeParameters,
    HANDGEN_OT_SaveHand,
    HANDGEN_OT_ResetJoints,
    HANDGEN_OT_RecordPose,
    HANDGEN_OT_RecordInitialPose,
    HANDGEN_OT_PlayReplay,
    HANDGEN_OT_AddToTimeline,
    HANDGEN_PT_MainPanel,
    HANDGEN_PT_PalmShape,
    HANDGEN_PT_FingersAndThumbs,
    HANDGEN_PT_HandGeometry,
    HANDGEN_PT_Options,
    HANDGEN_PT_JointControl,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.handgen_settings = bpy.props.PointerProperty(
        type=HANDGEN_PG_Settings)
    bpy.types.Scene.handgen_joints = bpy.props.CollectionProperty(
        type=HANDGEN_JointItem)
    bpy.types.Scene.handgen_replay_steps = bpy.props.IntProperty(
        name="Steps", default=30, min=2, max=300,
        description="Number of interpolation steps for replay")
    bpy.types.Scene.handgen_replay_duration = bpy.props.FloatProperty(
        name="Duration (s)", default=1.0, min=0.1, max=30.0,
        description="Duration of the animation in seconds")


def unregister():
    global _auto_update_timer
    if _auto_update_timer is not None:
        try:
            bpy.app.timers.unregister(_auto_update_timer)
        except Exception:
            pass
        _auto_update_timer = None

    for attr in (
        "handgen_replay_duration",
        "handgen_replay_steps",
        "handgen_joints",
        "handgen_settings",
    ):
        if hasattr(bpy.types.Scene, attr):
            delattr(bpy.types.Scene, attr)
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except RuntimeError:
            pass


if __name__ == "__main__":
    register()

