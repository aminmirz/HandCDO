"""A MANO-style human hand for the demonstration clip (Blender): one smooth skin surface in the light blue that
HaMeR / WiLoR use to render their MANO meshes. Right hand, same frame and API as glove_hand: palm in the XY plane,
fingers along +Y, palm (grasp) side towards +Z, thumb on the +X side.

The hand is built from anatomical parts (tapered palm, thenar and hypothenar pads, knuckles, phalanx capsules)
parented to joint empties; the parts drive contact tests and are fused into a single skin by a voxel remesh
(update_skin, called before each render). The MANO model files themselves are licence-restricted and not in
the repository.
build() returns (root, joints, digit_meshes, skin) with joints[name] = (empty, axis, rest, lower, upper).
"""
import math
import bmesh
import bpy
from mathutils import Matrix, Vector

SKIN_RGB = (.381, .517, .713)                               # HaMeR / WiLoR mesh colour LIGHT_BLUE (.651, .741, .859), linearised
FINGERS = [  # name, MCP x (m), MCP y (m), splay (deg), phalanx lengths (m), base radius (m)
    ('index', .027, .088, 4., (.041, .025, .019), .0090),
    ('middle', .0085, .092, 0., (.046, .029, .020), .0093),
    ('ring', -.0105, .088, -4., (.043, .027, .020), .0087),
    ('pinky', -.0275, .080, -9., (.034, .020, .018), .0076),
]
FLEX = ((90, 100, 75), (55, 65, 80))                        # finger, thumb flexion limits per joint (deg)
THUMB = dict(base=(.024, .016, .009), lengths=(.042, .032, .027), radius=.0118)
PALM = dict(w=.080, l=.088, t=.026)                         # width (x) at the knuckles, length (y), thickness (z)


def _material():
    m = bpy.data.materials.get('skin') or bpy.data.materials.new('skin')
    m.diffuse_color = (*SKIN_RGB, 1.); m.roughness = .55; return m


def _link(o, col):
    for c in o.users_collection: c.objects.unlink(o)
    col.objects.link(o)


def _capsule(name, r0, r1, length, parent, col):
    """Capsule along local +Y from 0 to length, radius r0 at the base tapering to r1, hemispherical ends."""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=14, radius=1.)
    o = bpy.context.object; o.name = name; _link(o, col)
    for v in o.data.vertices:
        y = v.co.y; r = r1 if y > 0 else r0
        v.co = Vector((v.co.x * r, y * r + (length if y > 0 else 0.), v.co.z * r * .93))
    o.parent = parent; o.matrix_parent_inverse = Matrix.Identity(4); o.matrix_basis = Matrix.Identity(4)
    return o


def _ellipsoid(name, centre, radii, parent, col):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=14, radius=1.)
    o = bpy.context.object; o.name = name; _link(o, col)
    o.data.transform(Matrix.Translation(centre) @ Matrix.Diagonal((*radii, 1.)))
    o.parent = parent; o.matrix_parent_inverse = Matrix.Identity(4); o.matrix_basis = Matrix.Identity(4)
    return o


def _empty(name, parent, col, matrix):
    e = bpy.data.objects.new(name, None); e.empty_display_size = .004; col.objects.link(e)
    e.parent = parent; e.matrix_parent_inverse = Matrix.Identity(4); e.matrix_basis = matrix; return e


def _palm(root, col):
    """Rounded slab, narrower and thicker at the heel than at the knuckles."""
    bpy.ops.mesh.primitive_cube_add(size=1.)
    o = bpy.context.object; o.name = 'hand_palm'; _link(o, col)
    w, l, t = PALM['w'], PALM['l'], PALM['t']
    for v in o.data.vertices:
        y = v.co.y + .5; x = v.co.x * w * (.80 + .20 * y); z = (v.co.z + .5) * t * (1.12 - .2 * y)
        v.co = Vector((x - .002 * (1 - y), y * l, z))
    bev = o.modifiers.new('round', 'BEVEL'); bev.width = .0105; bev.segments = 6
    sub = o.modifiers.new('smooth', 'SUBSURF'); sub.levels = sub.render_levels = 2
    o.parent = root; return o


def build():
    col = bpy.data.collections.new('hand_parts'); bpy.context.scene.collection.children.link(col)
    root = bpy.data.objects.new('hand_root', None); col.objects.link(root)
    palm = _palm(root, col); t = PALM['t']
    pads = [_ellipsoid('hand_wrist', (0, -.012, t * .52), (.029, .026, .017), root, col),                    # wrist (MANO cut)
            _ellipsoid('hand_hypothenar', (-.028, .036, t * .62), (.012, .034, .013), root, col),
            _ellipsoid('hand_thenar', (.020, .026, t * .66), (.019, .027, .014), root, col)]
    joints, digits = {}, {}
    for name, x, y, splay, lengths, r in FINGERS:
        parent = root; meshes = []
        m = Matrix.Translation((x, y, t * .52)) @ Matrix.Rotation(math.radians(-splay), 4, 'Z')
        pads.append(_ellipsoid(f'hand_{name}_knuckle', (x, y - .004, t * .55), (r * 1.15, r * 1.25, r * 1.2), root, col))
        for k, L in enumerate(lengths):
            e = _empty(f'hand_{name}_j{k}', parent, col, m)
            joints[f'{name}_{k}'] = (e, Vector((1, 0, 0)), m.copy(), 0., math.radians(FLEX[0][k]))
            r0 = r * (1 - .09 * k); r1 = r * (1 - .09 * (k + 1)) * (.92 if k == 2 else 1.)
            meshes.append(_capsule(f'hand_{name}_p{k}', r0, r1, L, e, col))
            parent = e; m = Matrix.Translation((0, L, 0))
        digits[name] = meshes
    # thumb: metacarpal from the heel of the palm, out to +X and forward, rolled towards the palm side; flexion
    # about local X sweeps it over the palm (+Z) and across a handle
    d = Vector((.78, .60, .12)).normalized(); ax = d.cross(Vector((0, 0, 1))).normalized(); z = ax.cross(d)
    m = Matrix.Translation(THUMB['base']) @ Matrix((ax, d, z)).transposed().to_4x4()
    parent = root; meshes = []; R = THUMB['radius']
    for k, L in enumerate(THUMB['lengths']):
        e = _empty(f'hand_thumb_j{k}', parent, col, m)
        joints[f'thumb_{k}'] = (e, Vector((1, 0, 0)), m.copy(), 0., math.radians(FLEX[1][k]))
        r0 = R * (1.15 if k == 0 else 1 - .07 * k); r1 = R * (1 - .07 * (k + 1)) * (.93 if k == 2 else 1.)
        meshes.append(_capsule(f'hand_thumb_p{k}', r0, r1, L, e, col))
        parent = e; m = Matrix.Translation((0, L, 0))
    digits['thumb'] = meshes
    digits['palm'] = [palm] + pads
    parts = [o for o in col.objects if o.type == 'MESH']
    for o in parts: o.hide_render = True; o.display_type = 'WIRE'
    skin = bpy.data.objects.new('hand_skin', bpy.data.meshes.new('hand_skin')); bpy.context.scene.collection.objects.link(skin)
    skin.data.materials.append(_material())
    rm = skin.modifiers.new('fuse', 'REMESH'); rm.mode = 'VOXEL'; rm.voxel_size = .0011; rm.use_smooth_shade = True
    sm = skin.modifiers.new('relax', 'CORRECTIVE_SMOOTH'); sm.iterations = 6; sm.use_only_smooth = True; sm.smooth_type = 'SIMPLE'
    skin['parts'] = [o.name for o in parts]
    bpy.context.view_layer.update(); update_skin(skin)
    return root, joints, digits, skin


def update_skin(skin):
    """Fuse the posed parts into the skin mesh (world space; the remesh modifier makes one smooth surface)."""
    bpy.context.view_layer.update(); dg = bpy.context.evaluated_depsgraph_get(); bm = bmesh.new()
    for n in skin['parts']:
        o = bpy.data.objects[n]; part = bmesh.new(); part.from_object(o, dg); part.transform(o.matrix_world)
        me = bpy.data.meshes.new('_tmp'); part.to_mesh(me); part.free(); bm.from_mesh(me); bpy.data.meshes.remove(me)
    skin.matrix_world = Matrix.Identity(4); bm.to_mesh(skin.data); bm.free(); skin.data.update()
    bpy.context.view_layer.update()


def pose(joints, fractions):
    """fractions: {digit: 0..1 of each joint's flexion range}."""
    for n, (e, axis, rest, lo, hi) in joints.items():
        f = fractions.get(n.rsplit('_', 1)[0], 0.)
        e.matrix_basis = rest @ Matrix.Rotation(lo + (hi - lo) * f, 4, axis)
    bpy.context.view_layer.update()
