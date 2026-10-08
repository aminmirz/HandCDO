"""Blender side: build creative_hands.DESIGNS with the native generator, assemble them with the add-on's own
pipeline (components.blend, materials, joint controls), export model.js for the v2 3D stage, and save a
Workbench preview in the approved style.

  blender -b --factory-startup --python-exit-code 1 --python export_creative_hands.py -- --ids compact duo
Run through build_creative_hands.py.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse, json, os, tempfile, time, zipfile
from render_thumb_orbit import ROOT, GEN, addon, style, assembly  # native add-on, approved style, assembly module
import bpy
from mathutils import Vector
from creative_hands import DESIGNS, build
from export_v2_model import export_scene

OUT = ROOT / 'v2/models'


def assemble(design, temp):
    hand, cfg = build(design, temp)
    data = hand.save_assembly_data(); palm = hand.save_palm_data()
    assert data and palm, design
    addon._save_hand_configs(hand, str(temp))
    bpy.context.scene['handgen_last_palm_npz'] = palm
    addon._clear_generated_hand(); addon._clear_template_objects()
    appended = addon._append_all_from_blend(str(GEN / 'blender/components.blend'))
    assembly.FingerAssembly.from_existing_scene(data, appended).run_assembly()
    assembly.PalmMesh(palm).generate(); assembly.assign_viz_materials(); assembly.apply_origin_offset(palm)
    addon._clear_template_objects()
    addon._setup_joint_controls(bpy.context, assembly, data)       # same joint table the add-on builds
    assembly.shade_all_meshes_flat(); addon._apply_hand_visibility(bpy.context)
    outline = cfg.outline_points
    return dict(palm_extent_mm=[float(v) for v in (outline.max(0) - outline.min(0))],
                finger_bases_mm=cfg.finger_bases.tolist(), finger_normals=cfg.finger_bases_normal_vectors.tolist(),
                finger_codes=[f.cfg.code for f in hand.fingers], thumb_codes=[t.cfg.code for t in hand.thumbs])


def preview(path):
    scene = bpy.context.scene; style(scene)
    visible = [o for o in scene.objects if o.type == 'MESH' and o.name.lower().startswith('viz_') and not o.hide_render]
    pts = [o.matrix_world @ Vector(c) for o in visible for c in o.bound_box]
    lo = Vector([min(p[i] for p in pts) for i in range(3)]); hi = Vector([max(p[i] for p in pts) for i in range(3)])
    centre = (lo + hi) / 2
    scene.camera.location = (centre.x, centre.y, 1); scene.camera.data.ortho_scale = max(hi.x - lo.x, (hi.y - lo.y) * 1.2) * 1.15
    scene.render.filepath = str(path); bpy.ops.render.render(write_still=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--ids', nargs='+', required=True)
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    addon.register(); s = bpy.context.scene.handgen_settings; s.auto_update = False
    s.generation_v2_dir = str(GEN); s.generate_collision_mesh = False
    for design in args.ids:
        start = time.monotonic(); out = OUT / design; temp = out / 'generation'; temp.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(temp); os.environ['TEMP'] = os.environ['TMP'] = str(temp)
        bpy.context.preferences.filepaths.temporary_directory = str(temp)
        s.save_output_dir = str(temp)
        label, palm, fingers, thumb = DESIGNS[design]
        info = assemble(design, temp)
        from motion_limits import limits
        motion = limits()                                            # collision-checked Flex / Spread ranges for the viewer
        print('MOTION', design, motion['flex'], motion['spread'], flush=True)
        spec = dict(label=label, palm=dict(size_mm=palm[0], sides=palm[1], aspect=palm[2], rotation_deg=palm[3]),
                    fingers=[dict(location=f[0], code=f[1], link_added_mm=f[2], angle_deg=f[3]) for f in fingers],
                    thumb=dict(location=thumb[0], code=thumb[1], link_added_mm=thumb[2]) if thumb else None,
                    finger_number=len(fingers), thumb_number=1 if thumb else 0, palm_size_mm=palm[0], motion=motion, **info)
        export_scene(out, design, spec)
        (out / 'design.json').write_text(json.dumps(spec, indent=2))
        preview(out / 'preview.png')
        with zipfile.ZipFile(out / 'configuration.zip', 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(temp.rglob('*')):
                if p.suffix in ('.py', '.npz'): z.write(p, p.name)
            z.write(out / 'design.json', 'design.json'); z.write(out / 'metadata.json', 'metadata.json')
        print(f'CREATIVE_EXPORT {design} {time.monotonic() - start:.1f}s', flush=True)


if __name__ == '__main__':
    main()
