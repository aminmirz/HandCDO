"""Render the co-design loop clips (generation_v2 hand from components.blend) in the approved style (Workbench, mylight.sl, Both cavity, white background):

  task    tools, then the high-score hand holding the hammer along the tracked demonstration trajectory (path drawn)
  close   joint-space sampling (D) then closing into the grasp (E): the exact hand, hammer, view and final grasp of the
          paper's grasp.mp4 (hand anime.blend; camera and grasp fitted to the clip, scripts/data/anime_grasp.json)
  wrench  wrench-space test (F) on that grasp: forces and torques along / about X, Y, Z applied to the tool, which moves in the grasp

  blender -b --factory-startup --python-exit-code 1 --python render_framework_clips.py -- --clip task|close|wrench [--still]
Frames go to website_assets/preview/framework-clips/<clip>/; build_framework_clips.py encodes them.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse, math, random
import bpy, bmesh
from mathutils import Matrix, Vector, Quaternion
from mathutils.bvhtree import BVHTree
import framework_scene as fs

OUT = fs.ROOT / 'preview/framework-clips'
HANDLE_R = .0155                                  # hammer handle radius (mesh x extent +/- 0.015-0.016 m)
GRIP = dict(y=.070, z=.035 + HANDLE_R, x=-.019, along=.21)   # handle axis across the fingers, resting on the palm


def clear():
    for o in list(bpy.data.objects): bpy.data.objects.remove(o)


def place_hammer(tool):
    """Handle along world +X across the fingers, grip point (handle-local z = GRIP.along) on the axis above the palm."""
    R = Matrix.Rotation(math.pi / 2, 4, 'Y') @ Matrix.Rotation(math.pi / 2, 4, 'Z')   # local +Z -> world +X, claw up
    tool.matrix_world = Matrix.Translation((GRIP['x'], GRIP['y'], GRIP['z'])) @ R @ Matrix.Translation((0, 0, -GRIP['along']))
    return tool.matrix_world.copy()


def tool_bvh(tool):
    bm = bmesh.new(); bm.from_mesh(tool.data); bm.transform(tool.matrix_world)
    t = BVHTree.FromBMesh(bm); bm.free(); return t


DIGIT_MESHES = None                                # set by the hand loader


def digit_meshes(digit):
    return DIGIT_MESHES(digit)


def touching(objs, bvh, clearance=.0015):
    for o in objs:
        mw = o.matrix_world; vs = o.data.vertices
        for k in range(0, len(vs), 3):
            loc, n, i, d = bvh.find_nearest(mw @ vs[k].co)
            if d is not None and d < clearance: return True
    return False


def contact_closure(joints, target, tool, steps=40):
    """Per digit, the largest fraction s of the simulated grasp angles before that digit touches the tool."""
    bvh = tool_bvh(tool); s_final = {}
    for digit in ('f1', 'f2', 'f3', 't1'):
        names = [n for n in target if n.startswith(f'J_{digit}_')]
        objs = digit_meshes(digit); s_ok = 0.
        for k in range(1, steps + 1):
            s = k / steps
            fs.pose(joints, {**{n: target[n] * s_final.get(n.split('_')[1], 0) for n in target}, **{n: target[n] * s for n in names}})
            bpy.context.view_layer.update()
            if touching(objs, bvh): break
            s_ok = s
        s_final[digit] = s_ok
    return s_final


def closed_angles(target, s_final, amount=1.):
    return {n: target[n] * s_final[n.split('_')[1]] * amount for n in target}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--clip', required=True); ap.add_argument('--still', action='store_true')
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    clear(); scene = bpy.context.scene
    if args.clip == 'task':                                           # human demonstration: MANO hand + tracked tools
        out = OUT / 'task'; out.mkdir(parents=True, exist_ok=True)
        import json
        labels = clip_demo_human(scene, out); (out / 'labels.json').write_text(json.dumps(labels)); print('FRAMES', len(labels), flush=True)
        return
    if args.clip in ('close', 'wrench', 'close_front', 'wrench_front'):   # the hand, hammer and view of grasp.mp4 (or a front view)
        out = OUT / args.clip; out.mkdir(parents=True, exist_ok=True)
        import json
        labels = anime_clip(scene, args.clip.split('_')[0], out, args.still, front=args.clip.endswith('_front')); (out / 'labels.json').write_text(json.dumps(labels)); print('FRAMES', len(labels), flush=True)
        return
    global DIGIT_MESHES
    root, joints, DIGIT_MESHES, grip = fs.load_v2_hand('compact')        # generation_v2 + components.blend, as the 3D viewer
    GRIP.update(grip); print('GRIP', {k: round(v, 4) for k, v in GRIP.items()}, flush=True)
    tool = fs.load_tool('hammer'); place_hammer(tool)
    cam = fs.setup(scene)                                               # after assembly, which clears scene objects
    target = fs.grasp_angles('hammer'); s_final = contact_closure(joints, target, tool)
    print('CLOSURE', {k: round(v, 3) for k, v in s_final.items()}, flush=True)
    out = OUT / args.clip; out.mkdir(parents=True, exist_ok=True)
    if args.still:
        fs.pose(joints, closed_angles(target, s_final)); bpy.context.view_layer.update()
        for name, d in (('front', (0, 0, 1)), ('side', (1, .15, .35)), ('iso', (.55, -.75, .6))):
            fs.look(cam, (0, .04, .03), d, .36); scene.render.filepath = str(out / f'still_{name}.png'); bpy.ops.render.render(write_still=True)
        return
    labels = {'close': clip_close, 'wrench': clip_wrench, 'task': clip_task}[args.clip](scene, cam, root, joints, tool, target, s_final, out)
    import json
    (out / 'labels.json').write_text(json.dumps(labels))
    print('FRAMES', len(labels), flush=True)


# ------------------------------------------------------------------ helpers
def material(name, rgb):
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.diffuse_color = (*rgb, 1.); return m


AXIS_RGB = {'x': (.84, .16, .16), 'y': (.18, .55, .23), 'z': (.18, .44, .84)}   # paper figure: X red, Y green, Z blue


def arrow(name, rgb, length=.07, radius=.0028):
    """Straight arrow along local +Z (shaft + cone), base at the origin."""
    bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=radius, depth=length * .78, location=(0, 0, length * .39))
    shaft = bpy.context.object
    bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=radius * 2.6, depth=length * .22, location=(0, 0, length * .89))
    head = bpy.context.object
    bpy.ops.object.select_all(action='DESELECT'); shaft.select_set(True); head.select_set(True)
    bpy.context.view_layer.objects.active = shaft; bpy.ops.object.join()
    shaft.name = name; shaft.data.materials.append(material(name, rgb)); return shaft


def ring_arrow(name, rgb, radius=.034, tube=.0026, sweep=1.6 * math.pi):
    """Curved arrow around local +Z (torque glyph)."""
    bm = bmesh.new(); seg, ring = 64, 10
    pts = []
    for i in range(seg + 1):
        a = sweep * i / seg; c = Vector((math.cos(a) * radius, math.sin(a) * radius, 0))
        t = Vector((-math.sin(a), math.cos(a), 0)); n = c.normalized(); b = Vector((0, 0, 1))
        pts.append([bm.verts.new(c + (n * math.cos(2 * math.pi * k / ring) + b * math.sin(2 * math.pi * k / ring)) * tube) for k in range(ring)])
    for i in range(seg):
        for k in range(ring):
            bm.faces.new((pts[i][k], pts[i][(k + 1) % ring], pts[i + 1][(k + 1) % ring], pts[i + 1][k]))
    me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
    obj = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(obj)
    a = sweep; tip = Vector((math.cos(a) * radius, math.sin(a) * radius, 0)); tan = Vector((-math.sin(a), math.cos(a), 0))
    bpy.ops.mesh.primitive_cone_add(vertices=20, radius1=tube * 2.8, depth=tube * 7, location=tip + tan * tube * 3.5)
    head = bpy.context.object; head.rotation_euler = tan.to_track_quat('Z', 'Y').to_euler()
    bpy.ops.object.select_all(action='DESELECT'); obj.select_set(True); head.select_set(True)
    bpy.context.view_layer.objects.active = obj; bpy.ops.object.join()
    obj.data.materials.append(material(name, rgb)); return obj


def tool_centre(tool):
    pts = [tool.matrix_world @ Vector(c) for c in tool.bound_box]
    return sum(pts, Vector()) / 8


def render(scene, path):
    scene.render.filepath = str(path); bpy.ops.render.render(write_still=True)


def ease(x):
    x = max(0., min(1., x)); return x * x * (3 - 2 * x)


# ------------------------------------------------------------------ D-E: joint-space sampling, then close on contact
def clip_close(scene, cam, root, joints, tool, target, s_final, out):
    fs.look(cam, (-.035, .045, .03), (.55, -.75, .6), .37)
    closed = closed_angles(target, s_final); rnd = random.Random(3)
    lim = {n: (j[3], j[4]) for n, j in joints.items()}
    samples = []
    for k in range(6):                                   # D: joint configurations sampled around the closing direction
        samples.append({n: max(lim[n][0], min(lim[n][1], closed[n] * rnd.uniform(.15, 1.05) + rnd.gauss(0, .12))) for n in closed})
    keys = [({}, 'Open hand')] + [(s, f'D · joint-space sample {i + 1}') for i, s in enumerate(samples)] + [({}, 'E · grasp initialization')]
    frames, labels = [], []
    hold, move = 9, 6
    for (a, la), (b, lb) in zip(keys, keys[1:]):
        for f in range(move): frames.append(({n: a.get(n, 0) + (b.get(n, 0) - a.get(n, 0)) * ease(f / move) for n in closed}, la if f < move // 2 else lb))
        for f in range(hold): frames.append((b, lb))
    for f in range(42):                                   # E: close smoothly until each digit touches the handle
        frames.append(({n: closed[n] * ease(f / 41) for n in closed}, 'E · grasp initialization'))
    frames += [(closed, 'E · grasp initialization')] * 24
    for i, (ang, lab) in enumerate(frames):
        fs.pose(joints, ang); bpy.context.view_layer.update(); render(scene, out / f'{i:04d}.png'); labels.append(lab)
    return labels


# ------------------------------------------------------------------ D-E-F on the grasp.mp4 hand (hand anime.blend)
CHAINS = ('finger_1', 'finger_2', 'finger_3', 'thumb_1')


FLEX = {'finger': {0: .35, 2: .6, 3: .6}, 'thumb': {2: .5, 3: .5}}   # extra flexion (fraction of upper limit) per joint index


def anime_motion(keys, joints):
    """Per joint frame: (start matrix, signed angle about the joint axis) of the closing motion: the recorded lean of
    grasp.mp4 (keys[0] -> keys[-1], mostly the H-Long joint) plus flexion of the bending joints, so digits wrap."""
    out = {}
    for n, m0 in keys[0].items():
        ax, a = (m0.inverted() @ keys[-1][n]).to_quaternion().to_axis_angle()
        e, axis, rest, lo, hi = joints[n]
        a = a * (1 if ax.dot(axis) >= 0 else -1) if a > 1e-4 else 0.
        kind = 'thumb' if '_thumb_' in n else 'finger'
        a += FLEX[kind].get(int(n.rsplit('_j', 1)[1]), 0.) * hi
        out[n] = (m0, a)
    return out


def overlaps(objs, tree):
    dg = bpy.context.evaluated_depsgraph_get()
    for o in objs:
        bm = bmesh.new(); bm.from_object(o, dg); bm.transform(o.matrix_world); t = BVHTree.FromBMesh(bm); bm.free()
        if t.overlap(tree): return True
    return False


def chain_of(n):
    return n.split('handgen_jfr_')[1].rsplit('_j', 1)[0]


def anime_pose(joints, motion, s, per_joint=None):
    """s: {chain: fraction of the closing motion}; per_joint optionally scales single joints (sampling). Angles are
    clamped to the joint limits."""
    for n, (e, axis, rest, lo, hi) in joints.items():
        m0, a = motion.get(n, (Matrix.Identity(4), 0.))
        ax0, b0 = m0.to_quaternion().to_axis_angle(); b0 = b0 if ax0.dot(axis) >= 0 else -b0
        k = s.get(chain_of(n), 0.) * (per_joint.get(n, 1.) if per_joint else 1.)
        e.matrix_basis = rest @ Matrix.Rotation(max(min(lo, b0), min(max(hi, b0), b0 + a * k)), 4, axis)
    bpy.context.view_layer.update()


def gap(objs, tree, every=6):
    """Smallest distance (m) from sampled vertices of objs to the tool surface."""
    d = 9.
    for o in objs:
        mw = o.matrix_world; vs = o.data.vertices
        for k in range(0, len(vs), every):
            r = tree.find_nearest(mw @ vs[k].co)
            if r[3] is not None: d = min(d, r[3])
    return d


def anime_closure(joints, motion, meshes, tool, top=3., steps=60):
    """Per chain, the largest fraction of the closing motion before that digit's meshes overlap the hammer; a digit
    that never touches stops where its tip comes closest to it."""
    tree = tool_bvh(tool); s_final = {}
    for c in CHAINS:
        ok, best = 0., (9., 0.)
        for k in range(1, steps + 1):
            s = top * k / steps; anime_pose(joints, motion, {**s_final, c: s})
            if overlaps(meshes(c), tree): break
            ok = s; best = min(best, (gap([o for o in meshes(c) if 'link_4' in o.name] or meshes(c), tree), s))
        else:
            ok = best[1]
        s_final[c] = ok
    return s_final


def anime_set(joints, deg):
    """deg: {joint frame name: angle in degrees about its local Z}; missing joints are zero."""
    for n, o in joints.items(): o.rotation_quaternion = Quaternion((0, 0, 1), math.radians(deg.get(n, 0.)))
    bpy.context.view_layer.update()


def keyed(keys, f):
    """Piecewise-linear joint angles at clip frame f between the fitted keys (as the original clip moves)."""
    for (f0, a0), (f1, a1) in zip(keys, keys[1:]):
        if f <= f1:
            t = (f - f0) / (f1 - f0); return {n: a0.get(n, 0.) + (a1.get(n, 0.) - a0.get(n, 0.)) * t for n in a1}
    return dict(keys[-1][1])


def no_penetration(joints, meshes, tree, ang, prev):
    """Pull each digit back toward its previous (contact-free) angles until it no longer overlaps the hammer."""
    ang = dict(ang)
    for c in CHAINS:
        names = [n for n in ang if f'_{c}_' in n]
        for _ in range(12):
            anime_set(joints, ang)
            if not overlaps(meshes(c), tree): break
            for n in names: ang[n] = prev.get(n, 0.) + (ang[n] - prev.get(n, 0.)) * .8
    return ang


# D: only the finger abduction joints (j1, +/-0.5 rad about the palm normal) and the thumb's first two joints are sampled
D_RANGE = {**{f'handgen_jfr_finger_{i}_j1': (-22., 22.) for i in (1, 2, 3)},
           'handgen_jfr_thumb_1_j0': (-75., 0.), 'handgen_jfr_thumb_1_j1': (-100., 20.)}   # degrees, inside the joint limits


def d_samples(joints, meshes, tree, open_, n=6):
    """Joint-space samples: D_RANGE joints drawn at random, the rest held open; no new digit-digit or digit-tool contact."""
    def bvh(c):
        dg = bpy.context.evaluated_depsgraph_get(); out = []
        for o in meshes(c):
            bm = bmesh.new(); bm.from_object(o, dg); bm.transform(o.matrix_world); out.append(BVHTree.FromBMesh(bm)); bm.free()
        return out
    def contacts():
        T = {c: bvh(c) for c in CHAINS}
        hits = {(c, d) for i, c in enumerate(CHAINS) for d in CHAINS[i + 1:] if any(a.overlap(b) for a in T[c] for b in T[d])}
        return hits | {(c, 'tool') for c in CHAINS if any(a.overlap(tree) for a in T[c])}
    anime_set(joints, open_); base = contacts()
    rnd = random.Random(3); out = []
    while len(out) < n:
        p = {**open_, **{k: rnd.uniform(lo, hi) for k, (lo, hi) in D_RANGE.items()}}
        anime_set(joints, p)
        if not contacts() - base: out.append(p)
    return out


def anime_clip(scene, clip, out, still=False, front=False):
    """D-E-F on the exact hand of grasp.mp4, moving through the joint keyframes fitted to that clip.
    front: look straight into the palm (hand +Z), hammer across the frame, instead of the grasp.mp4 view."""
    scene, joints, meshes, tool = fs.open_anime_exact()
    cam = fs.setup(scene, (1600, 900)); fs.set_anime_camera(cam)          # 16:9 as grasp.mp4
    if front:
        pts = [o.matrix_world @ Vector(c) for o in scene.objects if o.type == 'MESH' and o.visible_get() for c in o.bound_box]
        lo = Vector([min(p[k] for p in pts) for k in range(3)]); hi = Vector([max(p[k] for p in pts) for k in range(3)])
        cam.data.type = 'ORTHO'; cam.data.shift_x = cam.data.shift_y = 0.
        fs.look_roll(cam, (lo + hi) / 2, (0, 0, 1), 180, max(hi.x - lo.x, (hi.y - lo.y) * 16 / 9) * 1.4)   # rolled: fingers up
    keys = fs.anime_keys(); tree = tool_bvh(tool); open_, final = keys[0][1], keys[-1][1]
    path, prev = [], dict(open_)                         # E: the clip's 51 frames, contact-free
    for f in range(51):
        prev = no_penetration(joints, meshes, tree, keyed(keys, f), prev); path.append(prev)
    final = path[-1]
    if still:
        anime_set(joints, final); render(scene, out / 'still_closed.png'); anime_set(joints, open_); render(scene, out / 'still_open.png'); return []
    if clip == 'wrench':
        anime_set(joints, final); return wrench_frames(scene, tool, out, front)
    samples = d_samples(joints, meshes, tree, open_)
    anime_set(joints, open_)
    frames, labels = [], []
    seq = [(open_, 'Open hand')] + [(p, f'D · joint-space sample {i + 1}') for i, p in enumerate(samples)] + [(open_, 'E · grasp initialization')]
    hold, move = 9, 6
    for (a, la), (b, lb) in zip(seq, seq[1:]):
        for f in range(move + hold):
            t = ease(f / move) if f < move else 1.
            frames.append(({n: a.get(n, 0.) + (b.get(n, 0.) - a.get(n, 0.)) * t for n in b}, la if f < move // 2 else lb))
    frames += [(p, 'E · grasp initialization') for p in path] + [(final, 'E · grasp initialization')] * 24
    for i, (ang, lab) in enumerate(frames):
        anime_set(joints, ang); render(scene, out / f'{i:04d}.png'); labels.append(lab)
    return labels


# ------------------------------------------------------------------ F: wrench-space test
TESTS = [('f', ax, sg) for ax in 'xyz' for sg in (1, -1)] + [('t', ax, sg) for ax in 'xyz' for sg in (1, -1)]


def clip_wrench(scene, cam, root, joints, tool, target, s_final, out):
    closed = closed_angles(target, s_final); fs.pose(joints, closed); bpy.context.view_layer.update()
    fs.look(cam, (-.045, .045, .045), (.55, -.75, .6), .40)
    return wrench_frames(scene, tool, out)


def front_anchor(tool, hand_pt):
    """Centre of the tool's far end from the hand (the hammer head), and the tool's long axis pointing there."""
    V = [tool.matrix_world @ v.co for v in tool.data.vertices]
    c = sum(V, Vector()) / len(V); ax = max((Vector(e) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))), key=lambda e: max(abs((v - c).dot(e)) for v in V[::20]))
    if (c - hand_pt).dot(ax) < 0: ax = -ax
    top = max(v.dot(ax) for v in V); head = [v for v in V if v.dot(ax) > top - .045]
    return sum(head, Vector()) / len(head), ax, top


def wrench_frames(scene, tool, out, front=False):
    rest = tool.matrix_world.copy(); c = tool_centre(tool)
    glyphs = {}
    if front:                # glyphs on the hammer head, away from the hand; out-of-plane ones drawn obliquely
        hand_pt = bpy.data.objects['PalmBody_root'].matrix_world.translation
        A, out_ax, top = front_anchor(tool, hand_pt); view = Vector((0, 0, 1))
        obl = (out_ax + Vector((0, 0, 0)) - view * out_ax.dot(view)).normalized(); up = view.cross(obl)
        obl = (obl * .6 + up * .8).normalized()            # screen diagonal for the viewing axis: up and away from the hand
    for kind, ax, sg in TESTS:
        rgb = AXIS_RGB[ax]; name = f'{kind}{"+" if sg > 0 else "-"}{ax}'
        g = arrow(name, rgb) if kind == 'f' else ring_arrow(name, rgb)
        d = Vector({'x': (1, 0, 0), 'y': (0, 1, 0), 'z': (0, 0, 1)}[ax]) * sg
        if not front and kind == 'f':   # arrow pointing along the push direction, ending at the tool centre
            g.rotation_euler = d.to_track_quat('Z', 'Y').to_euler(); g.location = c - d * .11
        elif not front:                 # ring around the torque axis, turning in the torque's sense
            q = Vector((0, 0, 1)).rotation_difference(d); g.rotation_euler = q.to_euler(); g.location = c + d * .03
        elif kind == 'f':
            L = .07
            dd = d if abs(d.dot(view)) < .5 else (d * .35 + obl * (1 if d.dot(view) > 0 else -1)).normalized()
            g.rotation_euler = dd.to_track_quat('Z', 'Y').to_euler()
            if dd.dot(out_ax) < -.5: g.location = A + out_ax * (top - A.dot(out_ax) + .012) - dd * L   # pushes in from beyond the head end
            elif dd.dot(out_ax) > .5: g.location = A + out_ax * (top - A.dot(out_ax) + .012)          # pulls out from the head end
            else: g.location = A - dd * (L + .035)                                                    # onto the head from the side
        else:
            n = d if abs(d.dot(view)) > .5 else (d + view * 1.1 * (1 if d.dot(out_ax) >= 0 or ax != 'x' else 1)).normalized()
            q = Vector((0, 0, 1)).rotation_difference(n); g.rotation_euler = q.to_euler()
            g.location = A - out_ax * .05 if abs(d.dot(out_ax)) > .5 else A                           # about the handle: ring round it by the head
        g.show_in_front = True; g.hide_render = True; glyphs[name] = (g, kind, ax, d)
    if front:                # frame the hand, tool and every glyph
        bpy.context.view_layer.update()
        pts = [o.matrix_world @ Vector(cb) for o in scene.objects if o.type == 'MESH' and (o.visible_get() or o.name in glyphs) for cb in o.bound_box]
        lo = Vector([min(q[k] for q in pts) for k in range(3)]); hi = Vector([max(q[k] for q in pts) for k in range(3)])
        fs.look_roll(scene.camera, (lo + hi) / 2, (0, 0, 1), 180, max(hi.x - lo.x, (hi.y - lo.y) * 16 / 9) * 1.12)
    labels = []; per = 16; i = 0
    names = {'f': 'force', 't': 'torque'}
    for _ in range(6):                                    # settle
        render(scene, out / f'{i:04d}.png'); labels.append('F · grasp under test'); i += 1
    for kind, ax, sg in TESTS:
        name = f'{kind}{"+" if sg > 0 else "-"}{ax}'; g, _, _, d = glyphs[name]; g.hide_render = False
        for f in range(per):
            k = math.sin(math.pi * f / (per - 1))          # push, then release
            if kind == 'f':
                tool.matrix_world = Matrix.Translation(d * .006 * k) @ rest           # tool slips a few mm along the push
            else:
                tool.matrix_world = Matrix.Translation(c) @ Matrix.Rotation(math.radians(7) * k, 4, d) @ Matrix.Translation(-c) @ rest
            g.scale = (1, 1, 1) if kind == 't' else (1, 1, .6 + .4 * k)
            render(scene, out / f'{i:04d}.png'); labels.append(f'F · {names[kind]} {"+" if sg > 0 else "−"}{ax.upper()}'); i += 1
        g.hide_render = True; tool.matrix_world = rest
    return labels


# ------------------------------------------------------------------ task: tools, then the demonstrated trajectory
def smooth_poses(poses, w=5):
    import numpy as np
    P = np.array([[list(r) for r in m] for m in poses]); out = []
    for i in range(len(P)):
        a, b = max(0, i - w), min(len(P), i + w + 1)
        t = P[a:b, :3, 3].mean(0)
        qs = [Matrix(P[j][:3, :3].tolist()).to_quaternion() for j in range(a, b)]
        q = qs[0].copy()
        for k, qq in enumerate(qs[1:], 2):
            if q.dot(qq) < 0: qq = -qq
            q = q.slerp(qq, 1 / k)
        m = q.to_matrix().to_4x4(); m.translation = Vector(t.tolist()); out.append(m)
    return out


def clip_task(scene, cam, root, joints, tool, target, s_final, out):
    labels = []; i = 0
    # 1) the three task tools
    others = [fs.load_tool('stir'), fs.load_tool('knife')]
    hand_objs = [o for o in bpy.data.objects if o.name.startswith('viz_')]
    for o in hand_objs: o.hide_render = True
    lineup = [tool] + others; place = tool.matrix_world.copy()
    for k, t in enumerate(lineup):                     # long axis (local Z for all three scans) along world X, stacked
        pts = [Vector(c) for c in t.bound_box]; ctr = sum(pts, Vector()) / 8
        t.matrix_world = Matrix.Translation((0, (1 - k) * .085, 0)) @ Matrix.Rotation(math.pi / 2, 4, 'Y') @ Matrix.Translation(-ctr)
    fs.look(cam, (0, 0, 0), (0, -.35, 1), .40)
    for f in range(40):
        render(scene, out / f'{i:04d}.png'); labels.append('Task tools: hammer, spoon, knife'); i += 1
    for t in others: t.hide_render = True
    # 2) the hand holding the hammer along the tracked demonstration trajectory (camera frame -> Blender: flip y, z)
    for o in hand_objs: o.hide_render = False
    fs.pose(joints, closed_angles(target, s_final)); tool.matrix_world = place; bpy.context.view_layer.update()
    C = Matrix.Diagonal((1, -1, -1, 1)); M_place = place
    poses = smooth_poses(fs.trajectory('hammer'))[240:420]
    world_tool = [C @ p for p in poses]
    HEAD = Vector((0, 0, .03))                                            # tool-local point on the hammer head
    path_pts = [m @ HEAD for m in world_tool]
    import numpy as np
    P = np.array([list(p) for p in path_pts]); mu = P.mean(0)
    normal = Vector(np.linalg.svd(P - mu)[2][2].tolist())                 # swing plane normal (PCA)
    palm_dir = (world_tool[len(world_tool) // 2] @ M_place.inverted()).to_3x3() @ Vector((0, 0, 1))   # side the palm faces
    if normal.dot(palm_dir) < 0: normal = -normal
    grip = [m @ Vector((0, 0, GRIP['along'])) for m in world_tool]
    allp = np.array([list(p) for p in path_pts + grip]); span = float(np.ptp(allp, axis=0).max())
    view = (palm_dir.normalized() + normal).normalized()                  # between the palm side and the swing plane
    # frame the hand and the whole head path: project hand mesh bounds (mid frame) and the path onto the view plane
    root.matrix_world = world_tool[len(world_tool) // 2] @ M_place.inverted(); bpy.context.view_layer.update()
    hand_pts = [o.matrix_world @ Vector(c) for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith('viz_') for c in o.bound_box]
    up = Vector((0, 0, 1)) if abs(view.z) < .95 else Vector((0, 1, 0))
    rx = view.cross(up).normalized(); ry = rx.cross(view).normalized()
    pts = hand_pts + path_pts
    xs = [p.dot(rx) for p in pts]; ys = [p.dot(ry) for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    centre = rx * cx + ry * cy + view * (sum(p.dot(view) for p in pts) / len(pts))
    fs.look(cam, centre, view, max(max(xs) - min(xs), (max(ys) - min(ys)) * 1.2) * 1.12)
    print('VIEW', [round(v, 3) for v in view], 'normal', [round(v, 3) for v in normal], 'palm', [round(v, 3) for v in palm_dir], flush=True)
    trail_mat = material('trail', (1., .49, .30))
    curve = bpy.data.curves.new('trail', 'CURVE'); curve.dimensions = '3D'; curve.bevel_depth = .0022
    spl = curve.splines.new('POLY'); trail = bpy.data.objects.new('trail', curve); scene.collection.objects.link(trail)
    curve.materials.append(trail_mat)
    triad = [arrow(f'triad_{a}', AXIS_RGB[a], length=.05, radius=.0018) for a in 'xyz']
    base = {'x': Matrix.Rotation(math.pi / 2, 4, 'Y'), 'y': Matrix.Rotation(-math.pi / 2, 4, 'X'), 'z': Matrix.Identity(4)}
    for f, Tw in enumerate(world_tool):
        tool.matrix_world = Tw
        root.matrix_world = Tw @ M_place.inverted()                       # hand rigidly holds the tool
        n = f + 1; spl.points.add(max(0, n - len(spl.points)))
        for k in range(n): p = path_pts[k]; spl.points[k].co = (p.x, p.y, p.z, 1)
        trail.show_in_front = True
        for a, obj in zip('xyz', triad): obj.matrix_world = Tw @ Matrix.Translation(HEAD) @ base[a]; obj.show_in_front = True
        bpy.context.view_layer.update(); render(scene, out / f'{i:04d}.png'); labels.append('Tracked tool trajectory'); i += 1
    return labels



# ------------------------------------------------------------------ task: human demonstration with a glove hand
# per tool: handle radius (m), grip point along the tool's long axis (local z, m), traced tip point (local z, m),
# demonstration segment (frames of the tracked trajectory, 30 fps), label
DEMO = [('hammer', .0155, .23, .03, (240, 390), 'Hammering'),
        ('stir', .0095, .25, .05, (300, 450), 'Stirring'),
        ('knife', .0125, .285, .01, (150, 300), 'Cutting')]


def glove_closure(joints, digits, tool, steps=40):
    import glove_hand as gh
    bvh = tool_bvh(tool); dg = bpy.context.evaluated_depsgraph_get(); fr = {}
    def touch(objs):
        for o in objs:
            me = o.evaluated_get(dg).to_mesh()
            for k in range(0, len(me.vertices), 4):
                loc, n, i, d = bvh.find_nearest(o.matrix_world @ me.vertices[k].co)
                if d is not None and d < .0015: return True
        return False
    for d in ('index', 'middle', 'ring', 'pinky', 'thumb'):
        s = 0.
        for k in range(1, steps + 1):
            gh.pose(joints, {**fr, d: k / steps})
            if touch(digits[d]): break
            s = k / steps
        fr[d] = s
    gh.pose(joints, fr); return fr


_c, _s = math.cos(math.radians(30)), math.sin(math.radians(30))
GRIPS = {  # tool axes x, y, z in the hand frame (columns), and where the tool axis crosses the palm (m)
    'hammer': (((0, 0, -1), (_s, -_c, 0), (-_c, -_s, 0)), (0., .058)),   # power grip: handle diagonal, head out of the thumb side, face along the fingers
    'stir': (((0, 1, 0), (0, 0, 1), (1, 0, 0)), (0., .070)),          # fist: bowl out of the little-finger side
    'knife': (((0, 1, 0), (0, 0, -1), (-1, 0, 0)), (0., .070)),       # overhand: blade forward of the thumb side, edge down
}


def human_closure(joints, digits, tool, steps=80):
    """Close each digit until its parts overlap the tool (BVH test), then curl its outer joints on until they touch.
    Returns {joint name: fraction of its flexion range}."""
    import human_hand as hh
    tree = tool_bvh(tool); fr = {}
    def apply(fr):
        for n, (e, axis, rest, lo, hi) in joints.items():
            e.matrix_basis = rest @ Matrix.Rotation(lo + (hi - lo) * fr.get(n, 0.), 4, axis)
        bpy.context.view_layer.update()
    for d in ('index', 'middle', 'ring', 'pinky', 'thumb'):
        names = [n for n in joints if n.rsplit('_', 1)[0] == d]
        for stage in (names, names[1:], names[2:]):
            for k in range(1, steps + 1):
                trial = {**fr, **{n: min(1., fr.get(n, 0.) + 1. / steps) for n in stage}}
                apply(trial)
                if overlaps(digits[d], tree) or all(fr.get(n, 0.) >= 1. for n in stage): break
                fr = trial
    apply(fr); return fr


def place_props(task, world, tool, up=Vector((0, 0, 1))):
    """Ground / bowl / board for the task, at the lowest point the tool reaches (so it touches but never sinks in)."""
    import numpy as np
    V = [v.co.copy() for v in tool.data.vertices][::7]
    low = min((Tw @ v).z for Tw in world for v in V)
    path = [Tw.translation for Tw in world]; cx = sum(p.x for p in path) / len(path); cy = sum(p.y for p in path) / len(path)
    props = []
    def box(name, size, loc, rgb, rot=0.):
        bpy.ops.mesh.primitive_cube_add(size=1.); o = bpy.context.object; o.name = name
        o.scale = size; o.location = loc; o.rotation_euler = (0, 0, rot)
        bev = o.modifiers.new('round', 'BEVEL'); bev.width = .003; bev.segments = 3
        o.data.materials.append(material(name, rgb)); props.append(o); return o
    if task == 'hammer':                                  # plywood block on the floor, a nail at the strike point
        tipz = [(Tw @ v) for Tw in world for v in V]; strike = min(tipz, key=lambda p: p.z)
        box('plywood', (.30, .24, .04), (strike.x, strike.y, low - .02), (.86, .72, .52))
        bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=.0024, depth=.03, location=(strike.x, strike.y, low - .015))
        nail = bpy.context.object; nail.data.materials.append(material('nail', (.55, .56, .6))); props.append(nail)
    elif task == 'stir':                                  # bowl on the table around the stirring circle
        tips = [Tw @ Vector((0, 0, .05)) for Tw in world]
        mx = sum(p.x for p in tips) / len(tips); my = sum(p.y for p in tips) / len(tips); R = max(math.hypot(p.x - mx, p.y - my) for p in tips) + .035
        bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=1.); b = bpy.context.object; b.name = 'bowl'
        bm = bmesh.new(); bm.from_mesh(b.data); bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z > .02], context='VERTS')
        bm.to_mesh(b.data); bm.free(); b.scale = (R, R, R * .62); b.location = (mx, my, low - .004 + R * .62)
        so = b.modifiers.new('wall', 'SOLIDIFY'); so.thickness = .05; so.offset = 1.
        b.data.materials.append(material('bowl', (.93, .94, .96))); props.append(b)
    else:                                                 # cutting board on the table, along the stroke
        d = world[len(world) // 2].to_3x3() @ Vector((0, 0, 1)); rot = math.atan2(d.y, d.x)
        box('board', (.36, .19, .015), (cx, cy, low - .0075), (.90, .76, .56), rot)
    for o in props:
        for p in o.data.polygons: p.use_smooth = o.name in ('bowl', 'nail')
    return props


DEMO_SHOWN = ('hammer',)                          # tasks shown in the task clip (stir and knife are kept but not shown)
PROPS_SHOWN = ()                                  # tasks that get their ground / bowl / board (none: tool and hand only)


TRAIL_RGB, TRAIL_FADE = (.95, .25, .04), 60         # demonstration path colour; frames over which an old piece fades out


class fading_trail:
    """A smooth tube through the path points (Catmull-Rom, SUB pieces per step), newest end at full radius in TRAIL_RGB,
    blending to white and thinning to nothing TRAIL_FADE frames back (colour from a gradient texture along the tube)."""
    SUB, RING, R = 4, 12, .003

    def __init__(self, name, scene):
        img = bpy.data.images.new(name + '_grad', 256, 1)
        srgb = [c ** (1 / 2.2) for c in TRAIL_RGB]; px = []
        for k in range(256):
            t = k / 255; px += [c + (1 - c) * t for c in srgb] + [1.]
        img.pixels = px
        m = bpy.data.materials.new(name); m.use_nodes = True; nt = m.node_tree
        tex = nt.nodes.new('ShaderNodeTexImage'); tex.image = img; tex.interpolation = 'Linear'; tex.extension = 'EXTEND'
        bsdf = next(n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'); nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
        nt.nodes.active = tex; m.diffuse_color = (*TRAIL_RGB, 1.)
        self.me = bpy.data.meshes.new(name); self.me.materials.append(m)
        self.obj = bpy.data.objects.new(name, self.me); scene.collection.objects.link(self.obj); self.obj.show_in_front = True

    def update(self, pts):
        n = len(pts); ages = [(n - 1 - k) / TRAIL_FADE for k in range(n)]
        keep = [k for k in range(n) if ages[k] < 1.]
        if keep and keep[0] > 0: keep = [keep[0] - 1] + keep           # the piece that is fading out
        P = [pts[k] for k in keep]; A = [min(1., ages[k]) for k in keep]
        if len(P) < 2: self.obj.hide_render = True; return
        self.obj.hide_render = False
        C, CA = [], []                                                  # Catmull-Rom through the points
        for k in range(len(P) - 1):
            p0, p1, p2, p3 = P[max(k - 1, 0)], P[k], P[k + 1], P[min(k + 2, len(P) - 1)]
            for s_ in range(self.SUB):
                t = s_ / self.SUB
                C.append(.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
                CA.append(A[k] + (A[k + 1] - A[k]) * t)
        C.append(P[-1]); CA.append(A[-1])
        bm = bmesh.new(); uv = bm.loops.layers.uv.new('UVMap'); rings = []
        nrm = None
        for k, c in enumerate(C):
            tg = (C[min(k + 1, len(C) - 1)] - C[max(k - 1, 0)])
            tg = tg.normalized() if tg.length > 1e-9 else Vector((0, 0, 1))
            if nrm is None: nrm = tg.orthogonal().normalized()
            nrm = (nrm - tg * nrm.dot(tg)).normalized(); bin_ = tg.cross(nrm)       # parallel transport: no twisting
            r = self.R * max(0., 1 - CA[k]) ** .7
            rings.append([bm.verts.new(c + (nrm * math.cos(2 * math.pi * q / self.RING) + bin_ * math.sin(2 * math.pi * q / self.RING)) * r) for q in range(self.RING)])
        for k in range(len(rings) - 1):
            for q in range(self.RING):
                fc = bm.faces.new((rings[k][q], rings[k][(q + 1) % self.RING], rings[k + 1][(q + 1) % self.RING], rings[k + 1][q]))
                fc.smooth = True
                for lp in fc.loops:
                    kk = k if lp.vert in rings[k] else k + 1; lp[uv].uv = (min(.999, CA[kk]), .5)
        bm.to_mesh(self.me); bm.free(); self.me.update()


def clip_demo_human(scene, out):
    """The demonstration with the MANO hand (as HaMeR / WiLoR render it), each recording turned so gravity points down, with its ground,
    bowl or board, seen from the side that shows the motion (hammer swing, stirring circle, cutting stroke)."""
    import numpy as np, mano_hand as mh
    hand = mh.Hand(); root = hand.root; cam = fs.setup(scene, (1200, 900))
    tools = {t: fs.load_tool(t) for t, *_ in DEMO}
    trail_mat = material('trail', (1., .49, .30)); labels = []; i = 0
    G = Vector((0, -.6, -.8))                              # rough up in the (OpenCV) camera frame: these cameras look down
    for task, radius, along, tip_z, (f0, f1), name in [d for d in DEMO if d[0] in DEMO_SHOWN]:
        for t, o in tools.items(): o.hide_render = t != task
        tool = tools[task]; root.matrix_world = Matrix.Identity(4); hand.set({})
        cols, (gx, gy) = GRIPS[task]; R = Matrix([[cols[k][r] for k in range(3)] for r in range(3)]).to_4x4()
        gy *= hand.PALM['l'] / .088                       # grip positions were set on an 88 mm palm
        place = lambda dz: Matrix.Translation((gx, gy, hand.PALM['t'] + radius + dz)) @ R @ Matrix.Translation((0, 0, -along))
        M_place = place(mh.seat(hand, tool, place, tool_bvh)); tool.matrix_world = M_place; bpy.context.view_layer.update()
        fr = hand.close(tool_bvh(tool)); print('GRASP', task, {k: round(v, 2) for k, v in fr.items()}, flush=True)
        poses = smooth_poses(fs.trajectory(task), w=1)[f0:f1]               # 3-frame window: keeps the full swing (11 frames shrank it ~19%)
        TIP = Vector((0, 0, tip_z))
        tip = np.array([list(p @ TIP) for p in poses]); _, _, W = np.linalg.svd(tip - tip.mean(0))
        Rm = np.array([[list(r) for r in p.to_3x3()] for p in poses])
        if task == 'hammer': up = W[0]                     # the swing is up and down
        elif task == 'stir': up = W[2]                     # the stirring circle is level
        else:                                              # the blade stands upright, edge down (tool -y)
            x = Rm[:, :, 0].mean(0); y = Rm[:, :, 1].mean(0); up = y - x * (x @ y) / (x @ x)
        up = Vector((up / np.linalg.norm(up)).tolist())
        if up.dot(G) < 0: up = -up
        cx = Vector((1, 0, 0)); ex = (cx - up * cx.dot(up)).normalized(); ey = up.cross(ex)
        W2C = Matrix((ex, ey, up)).to_4x4()                # camera frame -> world with +Z up
        world = [W2C @ p for p in poses]; path = [m @ TIP for m in world]
        to_cam = -(W2C @ Vector(tip.mean(0).tolist()))     # towards the recording camera
        if task == 'hammer':                               # side-on to the swing: across the handle, level
            n = Vector((0, 0, 1)).cross(W2C.to_3x3() @ Vector(Rm[:, :, 2].mean(0).tolist())); elev = 12
        elif task == 'knife':                              # side-on to the blade
            n = W2C.to_3x3() @ Vector(Rm[:, :, 0].mean(0).tolist()); elev = 32
        else:                                              # from the front, above the bowl
            n = to_cam.copy(); elev = 38
        n.z = 0; n.normalize()
        if n.dot(to_cam) < 0: n = -n
        view = (n * math.cos(math.radians(elev)) + Vector((0, 0, math.sin(math.radians(elev))))).normalized()
        props = place_props(task, world, tool) if task in PROPS_SHOWN else []
        mid = world[len(world) // 2]; root.matrix_world = mid @ M_place.inverted(); tool.matrix_world = mid
        bpy.context.view_layer.update()
        pts = []                                           # frame the hand and tool over the whole segment, not one pose
        for Tw in world[::3]:
            tool.matrix_world = Tw; root.matrix_world = Tw @ M_place.inverted(); bpy.context.view_layer.update()
            pts += [o.matrix_world @ Vector(c) for o in (tool, hand.obj) for c in o.bound_box]
        pts += path + [p.matrix_world @ Vector(c) for p in props if p.name not in ('ground', 'table') for c in p.bound_box]
        rx = view.cross(Vector((0, 0, 1))).normalized(); ry = rx.cross(view).normalized()
        xs = [p.dot(rx) for p in pts]; ys = [p.dot(ry) for p in pts]
        centre = rx * (min(xs) + max(xs)) / 2 + ry * (min(ys) + max(ys)) / 2 + view * (sum(p.dot(view) for p in pts) / len(pts))
        fs.look(cam, centre, view, max(max(xs) - min(xs), (max(ys) - min(ys)) * 4 / 3) * 1.12)
        trail = fading_trail(f'trail_{task}', scene)       # one smooth tube along the path, fading with age
        triad = [arrow(f'triad_{task}_{a}', AXIS_RGB[a], length=.045, radius=.0017) for a in 'xyz']
        base = {'x': Matrix.Rotation(math.pi / 2, 4, 'Y'), 'y': Matrix.Rotation(-math.pi / 2, 4, 'X'), 'z': Matrix.Identity(4)}
        for f, Tw in enumerate(world):
            tool.matrix_world = Tw; root.matrix_world = Tw @ M_place.inverted()        # the hand holds the tool rigidly
            trail.update(path[:f + 1])
            for a, obj in zip('xyz', triad): obj.matrix_world = Tw @ Matrix.Translation(TIP) @ base[a]; obj.show_in_front = True
            render(scene, out / f'{i:04d}.png'); labels.append(f'{name} · tracked tool trajectory'); i += 1
        for o in triad + props + [trail.obj]: o.hide_render = True
    return labels


def clip_demo(scene, out):
    import numpy as np, glove_hand as gh
    root, joints, digits = gh.build(); cam = fs.setup(scene)
    tools = {t: fs.load_tool(t) for t, *_ in DEMO}
    C = Matrix.Diagonal((1, -1, -1, 1)); R = Matrix.Rotation(math.pi / 2, 4, 'Y') @ Matrix.Rotation(math.pi / 2, 4, 'Z')
    trail_mat = material('trail', (1., .49, .30)); labels = []; i = 0
    for task, radius, along, tip_z, (f0, f1), name in DEMO:
        for t, o in tools.items(): o.hide_render = t != task
        tool = tools[task]; root.matrix_world = Matrix.Identity(4); gh.pose(joints, {})
        # grip: handle across the fingers, resting on the palm (tool long axis -> hand +X)
        M_place = Matrix.Translation((0., .076, gh.PALM['t'] + radius)) @ R @ Matrix.Translation((0, 0, -along))
        tool.matrix_world = M_place; bpy.context.view_layer.update()
        fr = glove_closure(joints, digits, tool); print('GRASP', task, fr, flush=True)
        poses = smooth_poses(fs.trajectory(task))[f0:f1]; world = [C @ p for p in poses]
        TIP = Vector((0, 0, tip_z)); path = [m @ TIP for m in world]
        # view: between the palm side and the motion plane, framing the hand and the whole path
        P = np.array([list(p) for p in path]); normal = Vector(np.linalg.svd(P - P.mean(0))[2][2].tolist())
        palm_dir = (world[len(world) // 2] @ M_place.inverted()).to_3x3() @ Vector((0, 0, 1))
        if normal.dot(palm_dir) < 0: normal = -normal
        view = (palm_dir.normalized() + normal).normalized()
        root.matrix_world = world[len(world) // 2] @ M_place.inverted(); tool.matrix_world = world[len(world) // 2]; bpy.context.view_layer.update()
        pts = [o.matrix_world @ Vector(c) for o in bpy.data.objects if o.type == 'MESH' and not o.hide_render for c in o.bound_box] + path
        up = Vector((0, 0, 1)) if abs(view.z) < .95 else Vector((0, 1, 0)); rx = view.cross(up).normalized(); ry = rx.cross(view).normalized()
        xs = [p.dot(rx) for p in pts]; ys = [p.dot(ry) for p in pts]
        centre = rx * (min(xs) + max(xs)) / 2 + ry * (min(ys) + max(ys)) / 2 + view * (sum(p.dot(view) for p in pts) / len(pts))
        fs.look(cam, centre, view, max(max(xs) - min(xs), (max(ys) - min(ys)) * 1.2) * 1.15)
        curve = bpy.data.curves.new(f'trail_{task}', 'CURVE'); curve.dimensions = '3D'; curve.bevel_depth = .0022
        spl = curve.splines.new('POLY'); trail = bpy.data.objects.new(f'trail_{task}', curve); scene.collection.objects.link(trail)
        curve.materials.append(trail_mat); trail.show_in_front = True
        triad = [arrow(f'triad_{task}_{a}', AXIS_RGB[a], length=.045, radius=.0017) for a in 'xyz']
        base = {'x': Matrix.Rotation(math.pi / 2, 4, 'Y'), 'y': Matrix.Rotation(-math.pi / 2, 4, 'X'), 'z': Matrix.Identity(4)}
        for f, Tw in enumerate(world):
            tool.matrix_world = Tw; root.matrix_world = Tw @ M_place.inverted()        # the hand holds the tool rigidly
            n = f + 1; spl.points.add(max(0, n - len(spl.points)))
            for k in range(n): p = path[k]; spl.points[k].co = (p.x, p.y, p.z, 1)
            for a, obj in zip('xyz', triad): obj.matrix_world = Tw @ Matrix.Translation(TIP) @ base[a]; obj.show_in_front = True
            bpy.context.view_layer.update(); render(scene, out / f'{i:04d}.png'); labels.append(f'{name} · tracked tool trajectory'); i += 1
        trail.hide_render = True
        for o in triad: o.hide_render = True
    return labels


if __name__ == '__main__':
    main()
