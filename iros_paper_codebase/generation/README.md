# Generation

This is the original hand generation pipeline used for the IROS paper submission.

## Pipeline Overview

- `Config.py` defines palm, finger, thumb, hand, and plotting configuration values.
- `PalmClass.py` builds the palm outline, palm pad surface, and digit attachment locations.
- `FingerClass.py` builds finger and thumb segment geometry from code strings and link parameters.
- `HandClass.py` combines palm, finger, thumb, and hand-level configs into assembly data.
- `GenerateHands.py` is the script entry point for generating hand configurations and assembled outputs.
- `blender_full_assembly.py` performs the Blender-side assembly/export step.

## Assets

- `base_hand/` contains the baseline hand configuration and assembled hand asset used by this pipeline.
- `blender/components.blend` is the active Blender component library used by the assembly script.

## Generated Outputs

Generated hand folders contain per-hand config files, assembly metadata, Blender assembly outputs, and export data produced by the generation and Blender assembly scripts.
