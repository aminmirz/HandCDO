# IROS Paper Codebase

This folder groups the implementation used for the IROS paper submission.

## Structure

- `generation/`: original parametric hand generation pipeline used for the paper. It defines palm, finger, thumb, and hand configuration classes plus Blender assembly utilities.
- `optimization/`: nested design optimization code used for the paper results. It contains the V2 optimizer and the grasp evaluation pipeline it calls for scoring candidate hands.

## Flow

The paper codebase follows a generate-and-score loop. The generation code creates candidate hand geometries and exported models. The optimization code searches over design parameters, calls the generation pipeline for each candidate, and evaluates candidates through the grasp simulation code in `optimization/grasp_evaluation/`.
