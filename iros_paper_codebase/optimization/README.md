# Optimization

This folder contains the nested optimization code used for the paper results.

## Pipeline Overview

- `nested_optimizer_v2.py` is the optimizer used for the paper results.
- The optimizer samples hand design parameters, writes candidate hand configs, calls the generation code, and launches the nested grasp evaluation pipeline for scoring.
- `grasp_evaluation/` contains the Isaac Lab / RoboTouch simulation code used by the optimizer.

## Main Flow

1. Sample an outer hand design candidate.
2. Generate the corresponding hand model and URDF assets.
3. Run the inner grasp evaluation for the task objects.
4. Record scores and continue the nested search.

The default grasp simulation script is resolved relative to this folder as `grasp_evaluation/nested_multi_grasp_selector.py`.
