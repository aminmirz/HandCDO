from Config import *
import os
import subprocess
import numpy as np
from PalmClass import PalmOutline
from FingerClass import Finger
from HandClass import Hand

randomization_parameters = {
    'initial_finger_number': (1,4),
    'initial_thumb_number': 1,
    'initial_outline_sides': (4,6),
    'initial_outline_rotation_deg': 45.0,
    "initial_resolution_mm": 1.0,
    "initial_smoothing_iters": 6,
    "initial_smoothing_t": 0.25,

    "pad_resolution_level": 5,
    "initial_bump_type": 'gaussian',
    "initial_bump_number": (0, 3),
    "initial_bump_max_height_intensity_mm": (0.0, 15.0),
    "initial_bump_height_intensity": (0.0, 1.0),
    "initial_bumps_spread": (0.1, 0.5), # sigma scaled to the surface size
    "initial_bumps_aspect_ratio": (1.0, 3.0),
    "initial_bump_rotation_deg": (0, 180),
    "initial_bump_center_angle_deg": (0.0, 360.0), # bump center angle to x-axis
    "initial_bump_center_offset": (0.0, 1.0), # bump center offset from the origin of the surface outline scaled to the surface size
    "bump_height_intensity_list": [],
    "bumps_spread_list": [],
    "bumps_aspect_ratio_list": [],
    "bump_rotation_deg_list": [],
    "bump_center_angle_deg_list": [],
    "bump_center_offset_list": []

}

# finger_code_mode = 'random'
# thumb_code_mode = 'random'

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="generated_hands", help="Output directory for generated hands in separate directories.")
    parser.add_argument("--num_hands", type=int, default=10, help="Number of hands to generate.")


    args = parser.parse_args()

    for i in range(args.num_hands):
        output_dir = os.path.join(args.output_dir, f"hand_{i}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        palm_cfg = PalmConfig(data = 'fixed')
        palm_cfg.thumb_side = 'right'
        for key, value in randomization_parameters.items():
            setattr(palm_cfg, key, value)
        palm_cfg.generate_values()
        palm_cfg.update()
        palm_cfg.save_config(os.path.join(output_dir, "palm_cfg.py"))


        # finger_cfgs = [FingerConfig(data = finger_code_mode) for _ in range(palm_cfg.finger_number)]
        # thumb_cfgs = [FingerConfig(data = thumb_code_mode, type = 'thumb') for _ in range(palm_cfg.thumb_number)]
        finger_cfgs = [FingerConfig(data = 'random') for _ in range(palm_cfg.finger_number)]
        #change pad resolution level
        for finger_cfg in finger_cfgs:
            finger_cfg.pad_resolution_level = 4
        thumb_cfgs = [FingerConfig(data = 'random', type = 'thumb') for _ in range(palm_cfg.thumb_number)]
        for i, finger_cfg in enumerate(finger_cfgs):
            finger_cfg.save_config(os.path.join(output_dir, f"finger_cfg_{i}.py"))
        for i, thumb_cfg in enumerate(thumb_cfgs):
            thumb_cfg.save_config(os.path.join(output_dir, f"thumb_cfg_{i}.py"))
        hand_cfg = HandConfig(data = 'fixed')
        hand_cfg.collision_mesh = False
        hand_cfg.save_config(os.path.join(output_dir, "hand_cfg.py"))
        hand = Hand(palm_cfg=palm_cfg, finger_cfgs=finger_cfgs, thumb_cfgs=thumb_cfgs, hand_cfg=hand_cfg, root_dir=output_dir)
        hand.blender_full_assembly()