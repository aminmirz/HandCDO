"""Blender helpers for the framework clips: load a generated hand from its URDF export, pose it with simulated
joint angles, load task tools, and apply the approved render style (render_finger_location.style).

Imported by render_framework_clips.py inside Blender.
"""
import sys
sys.dont_write_bytecode = True
import math
from pathlib import Path
from xml.etree import ElementTree as ET
import bpy
from mathutils import Euler, Matrix, Vector

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from render_finger_location import ROOT, GEN, style   # approved Workbench + mylight.sl + Both cavity, white background

HCD = Path.home() / 'Desktop/PhD/_Projects/HandCoDesign'
HAND = HCD / 'experiments/study2/regenerated/#f3_high1_iter_170_hand_1/urdf_hand_export'   # paper's high-score design
JOINTS = HCD / 'experiments/joint_angles'                                                    # simulated grasps of that hand
TOOLS = ROOT.parent / 'iros_paper_codebase/optimization/grasp_evaluation/data'
PALETTE = {'f1': 'f1', 'f2': 'f2', 'f3': 'f3', 't1': 't'}   # digit -> components.blend material (f1 green, f2 yellow, f3 blue, t red)


def materials():
    names = ['palm', 'f1', 'f2', 'f3', 't', 'black', 'white_tip']
    missing = [n for n in names if bpy.data.materials.get(n) is None]
    if missing:
        with bpy.data.libraries.load(str(GEN / 'blender/components.blend'), link=False) as (src, dst):
            dst.materials = [n for n in missing if n in src.materials]
    return {n: bpy.data.materials.get(n) for n in names}


def material_for(mesh, link, mats):
    """Same convention as the generator's viz materials: palm crimson, covers/pads in the digit colour,
    fingertips white, motors / joints / link frames black."""
    digit = link.split('_')[1] if link.startswith('L_') else None
    if link == 'base_link':
        if 'PalmBody' in mesh: return mats['palm']
        if '_pad_f' in mesh: return mats[PALETTE[mesh.split('_pad_')[1][:2]]]
        if 'cover' in mesh: return mats['f1']
        return mats['black']
    if 'link_4' in mesh: return mats['white_tip']
    if 'cover' in mesh or '_pad_' in mesh: return mats[PALETTE[digit]]
    return mats['black']


def origin(el):
    o = el.find('origin') if el is not None else None
    xyz = [float(v) for v in (o.get('xyz', '0 0 0') if o is not None else '0 0 0').split()]
    rpy = [float(v) for v in (o.get('rpy', '0 0 0') if o is not None else '0 0 0').split()]
    return Matrix.Translation(xyz) @ Euler(rpy, 'XYZ').to_matrix().to_4x4()


def load_hand(path=HAND, name='hand'):
    """Build the URDF's link tree as empties with STL visuals. Returns (root, joints) where joints maps a joint
    name to (empty, axis, rest_matrix, lower, upper)."""
    mats = materials(); robot = ET.parse(path / 'hand_robot.urdf').getroot()
    cache, links, joints = {}, {}, {}
    col = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(col)
    for link in robot.findall('link'):
        e = bpy.data.objects.new(link.get('name'), None); e.empty_display_size = .004; col.objects.link(e); links[link.get('name')] = e
        for k, vis in enumerate(link.findall('visual')):
            fn = vis.find('geometry/mesh').get('filename').split('/')[-1]
            if fn not in cache:
                bpy.ops.wm.stl_import(filepath=str(path / 'meshes' / fn)); src = bpy.context.selected_objects[0]
                cache[fn] = src.data; bpy.data.objects.remove(src)
            o = bpy.data.objects.new(f"viz_{link.get('name')}_{k}", cache[fn]); col.objects.link(o)
            o.parent = e; o.matrix_parent_inverse = Matrix.Identity(4); o.matrix_basis = origin(vis)
            # meshes are shared between links, so the material is linked per object, not per mesh
            if len(o.data.materials) == 0: o.data.materials.append(None)
            o.material_slots[0].link = 'OBJECT'; o.material_slots[0].material = material_for(fn, link.get('name'), mats)
            for p in o.data.polygons: p.use_smooth = False
    for j in robot.findall('joint'):
        child = links[j.find('child').get('link')]; parent = links[j.find('parent').get('link')]
        child.parent = parent; child.matrix_parent_inverse = Matrix.Identity(4)
        rest = origin(j); child.matrix_basis = rest
        lim = j.find('limit'); axis = Vector([float(v) for v in j.find('axis').get('xyz').split()]).normalized()
        joints[j.get('name')] = (child, axis, rest, float(lim.get('lower')), float(lim.get('upper')))
    return links['base_link'], joints


def load_v2_hand(design='compact'):
    """Build a hand with the generation_v2 pipeline and generation_v2/blender/components.blend (same path as the
    3D viewer exports), with the add-on's own joint table. Returns (root, joints, digit_meshes, grip)."""
    import tempfile
    from render_thumb_orbit import addon
    import export_creative_hands as ec
    addon.register(); s = bpy.context.scene.handgen_settings; s.auto_update = False
    s.generation_v2_dir = str(GEN); s.generate_collision_mesh = False
    temp = ROOT / '.runtime/v2hand'; temp.mkdir(parents=True, exist_ok=True); tempfile.tempdir = str(temp); s.save_output_dir = str(temp)
    ec.assemble(design, temp)
    root = bpy.data.objects.new('hand_root', None); bpy.context.scene.collection.objects.link(root)
    for o in list(bpy.data.objects):
        if o.parent is None and o is not root and o.type in ('MESH', 'EMPTY'):
            o.parent = root; o.matrix_parent_inverse = Matrix.Identity(4)
    joints = {}
    for j in bpy.context.scene.handgen_joints:
        e = bpy.data.objects[j.link_frame_name]
        joints[j.name] = (e, Vector((j.axis_x, j.axis_y, j.axis_z)).normalized(), e.matrix_basis.copy(), j.min_val, j.max_val)
    cols = {'f1': 'finger_1', 'f2': 'finger_2', 'f3': 'finger_3', 't1': 'thumb_1'}
    digit_meshes = lambda d: [o for o in bpy.data.collections[cols[d]].all_objects if o.type == 'MESH' and not o.hide_render]
    bpy.context.view_layer.update()
    palm = [o for o in bpy.data.objects if o.type == 'MESH' and 'palm' in o.name.lower() and not o.hide_render]
    top = max((o.matrix_world @ Vector(c)).z for o in palm for c in o.bound_box)
    base = [bpy.data.objects[f'handgen_jfr_{cols[d]}_j0'].matrix_world.translation for d in ('f1', 'f2', 'f3')]
    grip = dict(x=sum(b.x for b in base) / 3, y=base[0].y - .004, z=top + .0155 + .002)   # handle across the fingers, on the palm
    return root, joints, digit_meshes, grip


ANIME = HCD / 'media/teaser_handgen/hand anime.blend'                   # source of the paper's grasp clip (grasp.mp4)
ANIME_TEX = HCD / 'HandGeneration/evaluation/grasp_sim/data/hammer/obj/material_0.png'   # hammer scan texture
ANIME_DATA = SCRIPTS / 'data/anime.json'      # its add-on settings, joint keyframes and hammer pose (extract_anime.py)
ANIME_VIEW = dict(direction=(-0.37776, -0.69438, 0.61248), roll=-13.)   # matched to grasp.mp4 by colour-mask search


def load_anime_hand():
    """The hand of grasp.mp4, regenerated with the current add-on and components.blend from the settings stored in
    hand anime.blend, plus its textured hammer at the stored palm-frame pose. Returns (root, joints, digit_meshes,
    tool, keyframes), keyframes = per frame {joint frame name: matrix_basis} of the original closing animation."""
    import json, tempfile
    from render_thumb_orbit import addon
    data = json.loads(ANIME_DATA.read_text())
    addon.register(); s = bpy.context.scene.handgen_settings; s.auto_update = False
    for k, v in data['settings'].items():
        if k == 'auto_update': continue
        try: s[k] = v
        except Exception as e: print('SETTING', k, e)
    s.generation_v2_dir = str(GEN); s.generate_collision_mesh = False
    temp = ROOT / '.runtime/v2hand'; temp.mkdir(parents=True, exist_ok=True); tempfile.tempdir = str(temp); s.save_output_dir = str(temp)
    assert bpy.ops.handgen.generate() == {'FINISHED'}
    root = bpy.data.objects.new('hand_root', None); bpy.context.scene.collection.objects.link(root)
    for o in list(bpy.data.objects):
        if o.parent is None and o is not root and o.type in ('MESH', 'EMPTY'):
            o.parent = root; o.matrix_parent_inverse = Matrix.Identity(4)
    joints = {}
    for j in bpy.context.scene.handgen_joints:
        e = bpy.data.objects[j.link_frame_name]
        joints[j.link_frame_name] = (e, Vector((j.axis_x, j.axis_y, j.axis_z)).normalized(), e.matrix_basis.copy(), j.min_val, j.max_val)
    cols = {'f1': 'finger_1', 'f2': 'finger_2', 'f3': 'finger_3', 't1': 'thumb_1'}
    digit_meshes = lambda d: [o for o in bpy.data.collections[cols[d]].all_objects if o.type == 'MESH' and not o.hide_render]
    with bpy.data.libraries.load(str(ANIME), link=False) as (src, dst):
        dst.objects = ['object']
    tool = dst.objects[0]; bpy.context.scene.collection.objects.link(tool); tool.name = 'tool_hammer'; tool.parent = None
    for im in bpy.data.images:
        if im.name.startswith('material_0'): im.filepath = str(ANIME_TEX); im.reload()
    bpy.context.view_layer.update()
    palm = bpy.data.objects.get('PalmBody_root'); P = palm.matrix_world.copy() if palm else Matrix.Identity(4)
    tool.matrix_world = P @ Matrix(data['frames'][0]['object_in_palm'])
    keys = [{n: Matrix(m) for n, m in f['joints'].items()} for f in data['frames']]
    return root, joints, digit_meshes, tool, keys


ANIME_GRASP = SCRIPTS / 'data/anime_grasp.json'   # camera and final grasp fitted to grasp.mp4 / graps/asd.png


ANIME_HAND = HCD / 'media/teaser_handgen/hand/assembled_hand_model.blend'   # the hand generated on 2026-03-06, used for grasp.mp4


def open_anime_exact():
    """Open the hand of grasp.mp4 itself (teaser_handgen/hand, generated 2026-03-06: flat palm, no pads; hand anime.blend
    is a later version with pads) and append the hammer from hand anime.blend at its stored palm-frame pose. Joint
    frames start at zero. Returns (scene, joints, digit_meshes, tool), joints[name] = joint frame (rotates about local Z)."""
    import json
    bpy.ops.wm.open_mainfile(filepath=str(ANIME_HAND))
    scene = bpy.context.scene
    scene.render.image_settings.media_type = 'IMAGE'          # the blend was saved with a movie output
    with bpy.data.libraries.load(str(ANIME), link=False) as (src, dst):
        dst.objects = ['object']
    tool = dst.objects[0]; scene.collection.objects.link(tool); tool.parent = None; tool.name = 'tool_hammer'
    for im in bpy.data.images:
        if im.name.startswith('material_0'): im.filepath = str(ANIME_TEX); im.reload()
    data = json.loads(ANIME_DATA.read_text())
    tool.matrix_world = bpy.data.objects['PalmBody_root'].matrix_world @ Matrix(data['frames'][0]['object_in_palm'])
    joints = {}
    for o in bpy.data.objects:
        if o.name.startswith('handgen_jfr_'):
            o.animation_data_clear(); o.rotation_mode = 'QUATERNION'; o.rotation_quaternion = (1, 0, 0, 0); joints[o.name] = o
    chain = lambda o: next((c for c in ('finger_1', 'finger_2', 'finger_3', 'thumb_1') if any(p.name.startswith(f'handgen_jfr_{c}_') for p in parents(o))), None)
    digit_meshes = lambda c: [o for o in scene.objects if o.type == 'MESH' and o.visible_get() and chain(o) == c]
    bpy.context.view_layer.update()
    return scene, joints, digit_meshes, tool


def parents(o):
    while o.parent is not None:
        o = o.parent; yield o


def set_anime_camera(cam):
    """The camera fitted to grasp.mp4 (perspective, 80 mm)."""
    import json
    c = json.loads(ANIME_GRASP.read_text())['camera']
    d = Vector((math.cos(c['el']) * math.cos(c['az']), math.cos(c['el']) * math.sin(c['az']), math.sin(c['el'])))
    look_roll(cam, c['centre'], d, c['roll'], c['scale'])
    cam.data.type = 'PERSP'; cam.data.lens = c['lens']; cam.data.sensor_fit = 'HORIZONTAL'; cam.data.sensor_width = 36
    cam.location = Vector(c['centre']) + d * (c['scale'] * c['lens'] / 36); cam.data.shift_x, cam.data.shift_y = c['shift_x'], c['shift_y']


def anime_keys():
    """[(clip frame, {joint frame: degrees})] fitted to grasp.mp4 frames 0, 10, ..., 50 (50 = graps/asd.png)."""
    import json
    return [(k['frame'], k['angles']) for k in json.loads(ANIME_GRASP.read_text())['keys']]


def pose_frames(joints, frame):
    """Apply one keyframe of the original animation (joint frame name -> matrix_basis), on top of each rest pose."""
    for n, (e, axis, rest, lo, hi) in joints.items():
        e.matrix_basis = rest @ frame.get(n, Matrix.Identity(4))


def look_roll(cam, target, direction, roll_deg, scale):
    """look() plus a roll of the camera about its viewing axis."""
    look(cam, target, direction, scale)
    q = cam.rotation_euler.to_quaternion() @ Matrix.Rotation(math.radians(roll_deg), 3, 'Z').to_quaternion()
    cam.rotation_euler = q.to_euler()


def pose(joints, angles):
    """angles: {joint name: radians}; missing joints stay at zero."""
    for n, (e, axis, rest, lo, hi) in joints.items():
        a = max(lo, min(hi, angles.get(n, 0.)))
        e.matrix_basis = rest @ Matrix.Rotation(a, 4, axis)


def grasp_angles(task):
    import numpy as np
    d = np.load(JOINTS / f'joint_angles_iter170_hand1_{task}.npz', allow_pickle=True)
    return {str(n): float(a) for n, a in zip(d['joint_names'], d['joint_angles'][0])}


def load_tool(task):
    folder = {'hammer': 'hammer', 'stir': 'spoon', 'knife': 'knife'}[task]
    obj_path = next((TOOLS / folder / 'obj').glob('*.obj'))
    bpy.ops.wm.obj_import(filepath=str(obj_path), forward_axis='Y', up_axis='Z')
    tool = bpy.context.selected_objects[0]; tool.name = f'tool_{task}'
    for p in tool.data.polygons: p.use_smooth = True
    return tool


def trajectory(task):
    """Recorded 6-D tool poses (3 x 4 per frame, camera frame, metres) from the task's tracked demonstration."""
    import numpy as np
    folder = TOOLS / {'hammer': 'hammer', 'stir': 'stir', 'knife': 'knife'}[task] / 'trajectories'
    out = []
    for f in sorted(folder.glob('*.txt')):
        a = np.loadtxt(f).reshape(-1, 4)[:3]                 # files hold 3 x 4 or 4 x 4 poses
        out.append(Matrix([[float(v) for v in r] for r in a] + [[0., 0., 0., 1.]]))
    return out


def setup(scene, resolution=(1200, 1000)):
    style(scene)                                        # approved look; replaces the camera with a front ortho one
    scene.display.shading.color_type = 'TEXTURE'        # tools keep their scanned textures; hand uses material colours
    scene.render.resolution_x, scene.render.resolution_y = resolution
    return scene.camera


def look(cam, target, direction, scale):
    """Orthographic camera looking at target from direction (unit vector), with ortho scale in metres."""
    target = Vector(target); d = Vector(direction).normalized()
    cam.location = target + d * 1.0
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    cam.data.ortho_scale = scale
