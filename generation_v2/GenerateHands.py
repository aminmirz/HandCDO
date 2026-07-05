from Config import HandConfig
from ParamGen import PalmParams, FingerParams
import os
from HandClass import Hand

# Range overrides for palm sampling. Only `initial_*` RANGE variables go here (they
# override PalmParams.set_ranges defaults). Non-range Config fields (e.g.
# pad_resolution_level) are set directly on the built config below.
palm_range_overrides = {
    'initial_finger_number': (1, 4),
    'initial_thumb_number': 1,
    'initial_outline_sides': (4, 6),
    'initial_outline_rotation_deg': 45.0,
    'initial_resolution_mm': 1.0,
    'initial_smoothing_iters': 6,
    'initial_smoothing_t': 0.25,
    'initial_bump_type': 'gaussian',
    'initial_bump_number': (0, 3),
    'initial_bump_max_height_intensity_mm': (0.0, 15.0),
    'initial_bump_height_intensity': (0.0, 1.0),
    'initial_bumps_spread': (0.1, 0.5),          # sigma scaled to the surface size
    'initial_bumps_aspect_ratio': (1.0, 3.0),
    'initial_bump_rotation_deg': (0, 180),
    'initial_bump_center_angle_deg': (0.0, 360.0),
    'initial_bump_center_offset': (0.0, 1.0),
}

# Non-range fixed Config fields set after building.
palm_pad_resolution_level = 5
finger_pad_resolution_level = 4

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="generated_hands", help="Output directory for generated hands in separate directories.")
    parser.add_argument("--num_hands", type=int, default=10, help="Number of hands to generate.")

    args = parser.parse_args()

    for hand_i in range(args.num_hands):
        output_dir = os.path.join(args.output_dir, f"hand_{hand_i}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # ---- palm: sample a config via ParamGen ----
        palm_cfg = PalmParams(fix=False, thumb_side='right', **palm_range_overrides).build()
        palm_cfg.pad_resolution_level = palm_pad_resolution_level
        palm_cfg.save_config(os.path.join(output_dir, "palm_cfg.py"))

        # ---- fingers / thumbs: sample configs via ParamGen ----
        finger_cfgs = [FingerParams('finger', fix=False).build() for _ in range(palm_cfg.finger_number)]
        for finger_cfg in finger_cfgs:
            finger_cfg.pad_resolution_level = finger_pad_resolution_level
        thumb_cfgs = [FingerParams('thumb', fix=False).build() for _ in range(palm_cfg.thumb_number)]

        for idx, finger_cfg in enumerate(finger_cfgs):
            finger_cfg.save_config(os.path.join(output_dir, f"finger_cfg_{idx}.py"))
        for idx, thumb_cfg in enumerate(thumb_cfgs):
            thumb_cfg.save_config(os.path.join(output_dir, f"thumb_cfg_{idx}.py"))

        hand_cfg = HandConfig(data='fixed')
        hand_cfg.collision_mesh = False
        hand_cfg.save_config(os.path.join(output_dir, "hand_cfg.py"))

        hand = Hand(palm_cfg=palm_cfg, finger_cfgs=finger_cfgs, thumb_cfgs=thumb_cfgs, hand_cfg=hand_cfg, root_dir=output_dir)
        hand.blender_full_assembly()
