"""Export one approval preview matching the author's existing Solid shading.

Run with Blender --background --factory-startup --python-exit-code 1 --python
this_script.py. Does not modify the website, animations, or original generator.
"""
import sys
sys.dont_write_bytecode = True
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=ROOT / 'site/models/baseline/assembled_hand_model.blend')
parser.add_argument('--output', type=Path, default=ROOT / 'preview/front-cavity')
parser.add_argument('--camera-roll', type=float, default=0., help='In-plane camera rotation in degrees; 90 makes the fingers point right.')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
OUT = args.output.resolve()
source = args.source.resolve()
if not OUT.is_relative_to(ROOT) or not source.is_relative_to(ROOT):
    raise ValueError('Preview source and output must remain inside website_assets.')
OUT.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(OUT)
os.environ['TEMP'] = os.environ['TMP'] = str(OUT)

bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=True)
bpy.context.preferences.filepaths.temporary_directory = str(OUT)
bpy.context.preferences.filepaths.save_version = 0
scene = bpy.context.scene

# The native model and all its material data are kept intact. Workbench displays
# the original materials' viewport colors instead of evaluating shader nodes.
visible = [obj for obj in scene.objects if obj.type == 'MESH'
           and obj.name.lower().startswith('viz_') and not obj.hide_render]
if not visible:
    raise RuntimeError('No generated hand meshes in the native scene.')
for obj in scene.objects:
    if obj.type == 'MESH' and obj not in visible:
        obj.hide_render = True

light = next((s for s in bpy.context.preferences.studio_lights if s.name == 'mylight.sl'), None)
if not light or not Path(light.path).is_file():
    raise RuntimeError('The reference mylight.sl studio light is missing.')
light_copy = OUT / 'mylight.sl'
shutil.copyfile(light.path, light_copy)

def configure(shading):
    shading.type = 'SOLID'
    shading.light = 'STUDIO'
    shading.studio_light = light.name
    shading.color_type = 'MATERIAL'
    shading.show_shadows = False
    shading.show_cavity = True
    shading.cavity_type = 'BOTH'
    # UI slider maxima, also matching both reference files. The World RNA hard
    # limit is 250, but its actual UI slider maximum is 2.5.
    for name in ['cavity_ridge_factor', 'cavity_valley_factor',
                 'curvature_ridge_factor', 'curvature_valley_factor']:
        setattr(shading, name, shading.bl_rna.properties[name].soft_max)
    shading.show_object_outline = True
    shading.object_outline_color = (0.0, 0.0, 0.0)
    shading.show_specular_highlight = True
    shading.background_type = 'VIEWPORT'
    shading.background_color = (1.0, 1.0, 1.0)
    shading.show_xray = False
    shading.studiolight_rotate_z = 0.0

scene.render.engine = 'BLENDER_WORKBENCH'
configure(scene.display.shading)
scene.display.render_aa = '32'
scene.render.resolution_x = 1440
scene.render.resolution_y = 1440
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.render.image_settings.color_mode = 'RGB'
scene.render.film_transparent = False
scene.render.dither_intensity = 0
scene.view_settings.view_transform = 'Standard'
scene.view_settings.look = 'None'
scene.view_settings.exposure = 0
scene.view_settings.gamma = 1

bpy.context.view_layer.update()
corners = [obj.matrix_world @ Vector(corner) for obj in visible for corner in obj.bound_box]
lo = Vector(tuple(min(p[i] for p in corners) for i in range(3)))
hi = Vector(tuple(max(p[i] for p in corners) for i in range(3)))
center = (lo + hi) / 2
camera_data = bpy.data.cameras.new('Front cavity preview camera')
camera = bpy.data.objects.new('Front cavity preview camera', camera_data)
scene.collection.objects.link(camera)
scene.camera = camera
camera.location = (center.x, center.y, hi.z + 1)
camera.rotation_euler = (0, 0, math.radians(args.camera_roll))  # Optical axis remains -Z.
camera_data.type = 'ORTHO'
view_rotation = camera.rotation_euler.to_matrix().transposed()
view_corners = [view_rotation @ (p-center) for p in corners]
camera_data.ortho_scale = max(max(p[i] for p in view_corners)-min(p[i] for p in view_corners) for i in (0,1)) * 1.18
camera_data.clip_start = .001
camera_data.clip_end = 10

for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            configure(area.spaces.active.shading)
            area.spaces.active.overlay.show_overlays = False
            area.spaces.active.region_3d.view_perspective = 'CAMERA'

image = OUT / 'front-cavity-preview.png'
scene.render.filepath = str(image)
bpy.context.view_layer.update()
from bpy_extras.object_utils import world_to_camera_view
projected = [world_to_camera_view(scene, camera, p) for p in corners]
bounds = [min(p.x for p in projected), min(p.y for p in projected),
          max(p.x for p in projected), max(p.y for p in projected)]
assert all(.02 < value < .98 for value in bounds), bounds
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / 'front-cavity-preview.blend'), copy=True)
bpy.ops.render.render(write_still=True)
report = {
    'source_model': str(source.relative_to(ROOT)),
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'engine': scene.render.engine,
    'reference_files': ['media/teaser_handgen/hand anime.blend', 'media/fingerGen_figure/fingers.blend'],
    'studio_light': 'mylight.sl',
    'studio_light_sha256': hashlib.sha256(light_copy.read_bytes()).hexdigest(),
    'cavity_type': 'BOTH', 'world_ridge': 2.5, 'world_valley': 2.5,
    'screen_ridge': 2.0, 'screen_valley': 2.0,
    'camera': 'Orthographic; front-on along -Z, perpendicular to the palm surface',
    'camera_roll_degrees': args.camera_roll,
    'background': 'White', 'color_source': 'Unchanged original material viewport colors',
    'resolution': [1440, 1440], 'visible_meshes': len(visible), 'projected_bounds': bounds,
    'approval': 'Preview only. No animations or webpage assets replaced.'
}
(OUT / 'settings.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
