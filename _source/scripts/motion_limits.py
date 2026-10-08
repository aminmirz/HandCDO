"""Collision-checked motion ranges for the v2 3D stage (run inside Blender on an assembled generation_v2 hand).

flex[chain]  fraction of each flexion joint's range (towards its larger-magnitude limit) a digit can close before
             any of its meshes newly overlaps a palm mesh (contacts already present at rest, e.g. mounts, are ignored);
             a thumb is further limited so that, with every finger at its own limit, it does not newly overlap a finger
spread       [inward, outward] fractions of the side joints' range, with every finger moved by the viewer's own
             side factor, before meshes of two different fingers newly overlap

The viewer maps Flex 100% and Spread +/-100% to these limits.
"""
import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

STEP, FINE = .05, .01


def _joints():
    out = []
    for j in bpy.context.scene.handgen_joints:
        lo, hi = j.min_val, j.max_val
        out.append(dict(name=j.name, chain=j.chain_id, obj=bpy.data.objects[j.link_frame_name], axis=Vector((j.axis_x, j.axis_y, j.axis_z)).normalized(),
                        lo=lo, hi=hi, sym=abs(lo + hi) < .05, target=hi if abs(hi) >= abs(lo) else lo))
    return out


def _side(joints):
    fingers = list(dict.fromkeys(j['chain'] for j in joints if j['chain'].startswith('finger')))   # same order as the viewer
    n = len(fingers)
    return {c: (i / (n - 1)) * 2 - 1 if n > 1 else 0. for i, c in enumerate(fingers)}


def _pose(joints, side, flex, spread):
    for j in joints:
        a = spread * side.get(j['chain'], 0.) * j['hi'] if j['sym'] else flex.get(j['chain'], 0.) * j['target']
        j['obj'].matrix_basis = Matrix.Rotation(max(j['lo'], min(j['hi'], a)), 4, j['axis'])
    bpy.context.view_layer.update()


def _bvh(objs):
    dg = bpy.context.evaluated_depsgraph_get(); out = {}
    for o in objs:
        ev = o.evaluated_get(dg); me = ev.to_mesh(); bm = bmesh.new(); bm.from_mesh(me); bm.transform(o.matrix_world)
        if len(bm.faces): out[o.name] = BVHTree.FromBMesh(bm)
        bm.free(); ev.to_mesh_clear()
    return out


def _hits(A, B, ignore):
    return {(a, b) for a, ta in A.items() for b, tb in B.items() if (a, b) not in ignore and ta.overlap(tb)}


def _scan(test, upto=1.):
    """Largest s in [0, upto] with test(s) False, coarse then fine (motion is monotone in practice)."""
    s = 0.
    while s + STEP <= upto + 1e-9 and not test(s + STEP): s += STEP
    while s + FINE <= upto + 1e-9 and not test(s + FINE): s += FINE
    return round(s, 3)


def limits():
    joints = _joints(); side = _side(joints); chains = list(dict.fromkeys(j['chain'] for j in joints))
    meshes = {c: [o for o in bpy.data.collections[c].all_objects if o.type == 'MESH' and not o.hide_render] for c in chains}
    in_chain = {o.name for ms in meshes.values() for o in ms}
    palm = [o for o in bpy.data.objects if o.type == 'MESH' and o.name.lower().startswith('viz_') and not o.hide_render and o.name not in in_chain]
    _pose(joints, side, {}, 0.); P = _bvh(palm)
    rest = {c: _bvh(meshes[c]) for c in chains}
    flex = {}
    for c in chains:                                         # close each digit until it newly touches the palm
        base = _hits(rest[c], P, set())
        def hit(s, c=c, base=base):
            _pose(joints, side, {c: s}, 0.); return bool(_hits(_bvh(meshes[c]), P, base))
        flex[c] = _scan(hit)
    fingers_all = [c for c in chains if c.startswith('finger')]
    for c in chains:                                         # thumbs: stop before running into the closed fingers
        if not c.startswith('thumb') or not fingers_all: continue
        base = {h for d in fingers_all for h in _hits(rest[c], rest[d], set())}
        def into(s, c=c, base=base):
            _pose(joints, side, {**{d: flex[d] for d in fingers_all}, c: s}, 0.); T = _bvh(meshes[c])
            return any(_hits(T, _bvh(meshes[d]), base) for d in fingers_all)
        flex[c] = max(0., round(_scan(into, flex[c]) - .05, 3))   # small margin: stop short of touching
    fingers = [c for c in chains if c.startswith('finger') and side.get(c)]
    movers = [c for c in chains if c.startswith('finger')]
    base = {(a, b) for i, c in enumerate(movers) for d in movers[i + 1:] for (a, b) in _hits(rest[c], rest[d], set())}
    def collide(s):
        _pose(joints, side, {}, s); T = {c: _bvh(meshes[c]) for c in movers}
        return any(_hits(T[c], T[d], base) for i, c in enumerate(movers) for d in movers[i + 1:])
    spread = [_scan(lambda s: collide(-s)), _scan(lambda s: collide(s))] if fingers else [0., 0.]
    _pose(joints, side, {}, 0.)
    return dict(flex=flex, spread=spread, method='BVH mesh overlap, contacts present at rest ignored', step=FINE)
