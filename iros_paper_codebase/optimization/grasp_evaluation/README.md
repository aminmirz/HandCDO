# Grasp Simulation

This folder contains the nested grasp simulation pipeline called by `../nested_optimizer_v2.py`.

## Pipeline Overview

- `nested_multi_grasp_selector.py`: launches Isaac Sim, evaluates generated hand URDFs across task objects, and writes per-hand scores.
- `nested_multi_sim_config.py`: defines the multi-hand, multi-task Isaac Lab simulation environment.
- `grasp_optimizer.py`: samples and tracks grasp parameters for the inner TPE optimization loop.
- `utils.py`: transformation utilities used by the task trajectory loader.
- `robotouch/`: URDF parsing, URDF padding, and asset/data path helpers used by the nested simulation.

## Assets And Data

- `assets/objects/` contains the task object USD assets used in simulation.
- `data/` contains task trajectory data used to evaluate object-specific grasps.
- `robotouch/paths.py` centralizes paths for assets, data, and generated URDF padding.

## Role In Optimization

The optimizer passes generated hand URDFs into this folder's simulation entry point. The simulation pipeline evaluates each candidate hand against the configured task objects and returns scores used by the outer optimizer.
