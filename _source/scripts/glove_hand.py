"""A simple, stylised glove hand for the demonstration clip (Blender). Right hand; same frame as the generated
robot hands: palm in the XY plane, fingers along +Y, palm (grasp) side towards +Z, thumb on the +X side.

Each phalanx is an ellipsoid parented to a joint empty; flexion is a rotation about the joint's local X axis.
build() returns (root, joints, digit_meshes) with joints[name] = (empty, axis, rest, lower, upper).
"""
import math
import bpy
from mathutils import Matrix, Vector

GLOVE_RGB = (.50, .72, .97)                                 # light blue, as the glove in the framework figure
FINGERS = [  # name, base x (m), phalanx lengths (m), radius (m)
    ('index', .029, (.044, .027, .021), .0088),
    ('middle', .0095, (.049, .031, .023), .0092),
    ('ring', -.0105, (.046, .029, .022), .0088),
    ('pinky', -.029, (.036, .022, .018), .0078),
]
THUMB = dict(base=(.040, .018, .010), lengths=(.042, .032, .026), radius=.0105)
PALM = dict(w=.086, l=.094, t=.026)                         # width (x), length (y), thickness (z)


def _material():
    m = bpy.data.materials.get('glove') or bpy.data.materials.new('glove')
    m.diffuse_color = (*GLOVE_RGB, 1.); m.roughness = .6; return m


def _ellipsoid(name, radius, length, parent, col, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=28, ring_count=16, radius=1.)
    o = bpy.context.object; o.name = name
    for c in o.users_collection: c.objects.unlink(o)
    col.objects.link(o)
    o.data.transform(Matrix.Diagonal((radius, length / 2 + radius * .55, radius * .92, 1.)))   # long along local +Y
    o.data.transform(Matrix.Translation((0, length / 2, 0)))
    o.parent = parent; o.matrix_parent_inverse = Matrix.Identity(4); o.matrix_basis = Matrix.Identity(4)
    o.data.materials.append(mat)
    for p in o.data.polygons: p.use_smooth = True
    return o


def _empty(name, parent, col, matrix):
    e = bpy.data.objects.new(name, None); e.empty_display_size = .004; col.objects.link(e)
    e.parent = parent; e.matrix_parent_inverse = Matrix.Identity(4); e.matrix_basis = matrix; return e


def build():
    mat = _material(); col = bpy.data.collections.new('glove'); bpy.context.scene.collection.children.link(col)
    root = bpy.data.objects.new('glove_root', None); col.objects.link(root)
    # palm: rounded box, bottom (back of the hand) at z = 0, palm surface at z = t
    bpy.ops.mesh.primitive_cube_add(size=1.)
    palm = bpy.context.object; palm.name = 'glove_palm'
    for c in palm.users_collection: c.objects.unlink(palm)
    col.objects.link(palm)
    palm.data.transform(Matrix.Diagonal((PALM['w'], PALM['l'], PALM['t'], 1.)))
    palm.data.transform(Matrix.Translation((0, PALM['l'] / 2, PALM['t'] / 2)))
    bev = palm.modifiers.new('round', 'BEVEL'); bev.width = .011; bev.segments = 6
    sub = palm.modifiers.new('smooth', 'SUBSURF'); sub.levels = 2; sub.render_levels = 2
    palm.parent = root; palm.data.materials.append(mat)
    for p in palm.data.polygons: p.use_smooth = True
    joints, digits = {}, {}
    for name, x, lengths, r in FINGERS:
        parent = root; m = Matrix.Translation((x, PALM['l'] - .004, PALM['t'] * .5)); meshes = []
        for k, L in enumerate(lengths):
            e = _empty(f'glove_{name}_j{k}', parent, col, m)
            joints[f'{name}_{k}'] = (e, Vector((1, 0, 0)), m.copy(), 0., math.radians((90, 100, 80)[k]))
            meshes.append(_ellipsoid(f'glove_{name}_p{k}', r * (1 - .07 * k), L, e, col, mat))
            parent = e; m = Matrix.Translation((0, L, 0))
        digits[name] = meshes
    # thumb: from the palm's +X side, pointing out and forward, rolled towards the palm side
    b = THUMB['base']; parent = root
    # explicit frame: local +Y = thumb direction (out to +X, forward, up); local X = bend axis chosen so that
    # bending sweeps the thumb up over the palm side (+Z) and across the handle
    d = Vector((.70, .55, .45)).normalized(); ax = d.cross(Vector((0, 0, 1))).normalized(); z = ax.cross(d)
    m = Matrix.Translation(b) @ Matrix((ax, d, z)).transposed().to_4x4()
    meshes = []
    for k, L in enumerate(THUMB['lengths']):
        e = _empty(f'glove_thumb_j{k}', parent, col, m)
        joints[f'thumb_{k}'] = (e, Vector((1, 0, 0)), m.copy(), 0., math.radians((55, 70, 80)[k]))
        meshes.append(_ellipsoid(f'glove_thumb_p{k}', THUMB['radius'] * (1 - .06 * k), L, e, col, mat))
        parent = e; m = Matrix.Translation((0, L, 0))
    digits['thumb'] = meshes
    digits['palm'] = [palm]
    bpy.context.view_layer.update()
    return root, joints, digits


def pose(joints, fractions):
    """fractions: {digit: 0..1 of each joint's flexion range}."""
    for n, (e, axis, rest, lo, hi) in joints.items():
        f = fractions.get(n.rsplit('_', 1)[0], 0.)
        e.matrix_basis = rest @ Matrix.Rotation(lo + (hi - lo) * f, 4, axis)
    bpy.context.view_layer.update()
