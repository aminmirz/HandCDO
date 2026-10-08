# Generation V2

`generation_v2/` is the improved hand generation pipeline and is kept at the root of the public repository to distinguish it from the original IROS paper generator.

## Pipeline Overview

- `ParamGen.py` samples concrete design parameters for the palm, fingers, thumbs, and surface pads.
- `Config.py` stores the config objects that can be saved to and loaded from per-hand Python config files.
- `PalmClass.py`, `FingerClass.py`, and `HandClass.py` build the geometry, attachment points, kinematic chains, and assembly metadata.
- `blender_full_assembly.py` converts generated geometry and assembly metadata into Blender objects, pads, joints, collision meshes, and exportable hand assets.

## Script Entry Points

- `GenerateHands.py`: generates `--num_hands` hand designs into `--output_dir`.
- `GenerateHandsBatch.py`: repeatedly generates hands until `--target_successes` is reached or `--max_attempts` is exhausted. It writes `batch_summary.json` with success and failure records.
- `ParamGen.py`: contains `PalmParams` and `FingerParams`, the main sampling builders used by both generation scripts.

Example:

```bash
python GenerateHands.py --output_dir generated_hands --num_hands 10
python GenerateHandsBatch.py --output_dir generated_hands_batch --target_successes 50 --max_attempts 80
```

## Main Parameter Groups

`PalmParams` controls the palm and digit placement:

- Palm outline: `initial_palm_size_mm`, `initial_resolution_mm`, `initial_outline_sides`, `initial_outline_aspect_ratio`, `initial_outline_longer_axis`, `initial_outline_rotation_deg`, `initial_smoothing_iters`, and `initial_smoothing_t`.
- Digit counts and placement: `initial_finger_number`, `initial_thumb_number`, `thumb_side`, `generation_mode`, per-digit base angles, normal offsets, side offsets, and location values along the valid palm outline curve.
- Palm surface pads: bump type, bump count, maximum height, intensity, spread, aspect ratio, rotation, center angle, and center offset.

The current script overrides in `GenerateHands.py` and `GenerateHandsBatch.py` sample 1-4 fingers, one right-side thumb, 4-6 palm outline sides, 1 mm outline resolution, Gaussian palm bumps, palm pad resolution level 5, and finger pad resolution level 4.

`FingerParams` controls each finger and thumb chain:

- Structural code format: fingers use `R-BEFORE-AFTER`, and thumbs use `R--AFTER`.
- Rotation mode: `R` selects the kinematic rotation pattern. Finger modes cover no rotation, vertical rotation, horizontal short/long rotation, and combined vertical/horizontal rotation. Thumb modes cover primary and primary-plus-secondary rotation.
- Joint digits: `1` represents a short added joint and `2` represents a long added joint.
- Link and pad sampling: added link lengths, per-link pad bump type, bump count, bump height, spread, aspect ratio, rotation, center angle, and center offset.

## Blender Assets And Add-On

- `blender/components.blend` is the active Blender component library used by the V2 assembly/export pipeline.
- `blender/HandGeneratorV2.py` is a Blender add-on that exposes palm shape, generation mode, finger/thumb counts, thumb side, hand geometry, pad settings, per-digit locations, and per-finger/per-thumb code controls in the Blender UI.
- `blender_full_assembly.py` is the script used by the Python generators and the add-on to assemble the final hand model.

## Generated Hand Outputs

A generated hand directory contains the sampled configs and assembly artifacts used by the Blender export step:

- `palm_cfg.py`
- `finger_cfg_*.py`
- `thumb_cfg_*.py`
- `hand_cfg.py`
- `assembly_data.npz`
- `palm_outlines_data.npz`
- `assembled_hand_model.blend`
- `urdf_hand_export/` when URDF export is run
