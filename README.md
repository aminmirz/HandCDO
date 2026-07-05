# HandCDO

HandCDO contains the public code and project page assets for task-driven robotic hand generation and co-design. The repository separates the IROS paper implementation from the newer generator: `iros_paper_codebase/` keeps the paper code together, while `generation_v2/` is the improved hand generation pipeline at the repository root.

## Repository Layout

```text
HandCDO/
  index.html                 # Static project page entry point
  website_assets/            # Images and video used by the project page
  generation_v2/             # Improved hand generation pipeline
  iros_paper_codebase/       # Codebase used for the IROS submission
    generation/              # Original hand generation pipeline
    optimization/            # Nested hand design optimization code
      grasp_evaluation/      # Isaac Lab / grasp simulation scoring pipeline
```

## Main Workflows

- Use `generation_v2/` for the current generator. It includes randomized parameter sampling, batch generation, Blender assembly/export code, and a Blender add-on under `generation_v2/blender/`.
- Use `iros_paper_codebase/generation/` for the original generator used in the IROS submission.
- Use `iros_paper_codebase/optimization/` for the nested optimization pipeline. The optimizer generates candidate hand designs and scores them through `iros_paper_codebase/optimization/grasp_evaluation/`.
- The project page is defined by `index.html` and the media in `website_assets/`.

## Code Organization

- `generation_v2/ParamGen.py` defines the current sampling layer for palm, finger, thumb, and pad parameters.
- `generation_v2/GenerateHands.py` and `generation_v2/GenerateHandsBatch.py` are the script entry points for generating one or more hand designs.
- `generation_v2/blender/HandGeneratorV2.py` is the Blender add-on entry point, and `generation_v2/blender/components.blend` stores reusable Blender component assets used by the assembly pipeline.
- `iros_paper_codebase/optimization/nested_optimizer_v2.py` is the optimizer script used for the reported paper results.

## License

This project is released under the MIT License. See `LICENSE`.
