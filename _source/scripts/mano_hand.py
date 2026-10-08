"""The MANO right hand (the mesh HaMeR and WiLoR regress) for the demonstration clip, posed in Blender with numpy.

Needs scripts/data/mano/models/mano_right.npz, converted from the licence-restricted MANO_RIGHT.pkl (not in the
repository; see README). Mean shape (betas = 0); linear blend skinning with MANO's pose correctives.

Hand frame (as glove_hand / human_hand): fingers along +Y (wrist -> middle knuckle), palm (grasp) side towards +Z,
thumb on the +X side, palm surface at z = PALM['t'].
"""
from pathlib import Path
import math
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

NPZ = Path(__file__).resolve().parent / 'data/mano/models/mano_right.npz'
SKIN_RGB = (.381, .517, .713)                               # HaMeR / WiLoR LIGHT_BLUE (.651, .741, .859), linearised
CHAINS = {'index': (1, 2, 3), 'middle': (4, 5, 6), 'pinky': (7, 8, 9), 'ring': (10, 11, 12), 'thumb': (13, 14, 15)}
FLEX = {'finger': (95, 105, 80), 'thumb': (55, 55, 70)}    # flexion limits per joint (deg); thumb base: lift off the palm


def _rodrigues(r):
    th = np.linalg.norm(r)
    if th < 1e-9: return np.eye(3)
    k = r / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


class Hand:
    def __init__(self):
        d = np.load(NPZ)
        self.v0, self.f, self.W, self.P = d['v_template'], d['f'], d['weights'], d['posedirs']
        self.J = d['J_regressor'] @ self.v0; self.parent = d['kintree'][0].copy(); self.parent[0] = -1
        self.owner = self.W.argmax(1)                         # dominant joint per vertex
        # frame: fingers (wrist -> middle MCP), thumb side, palm normal = thumb side x fingers (right hand)
        J = self.J; y = J[4] - J[0]; y /= np.linalg.norm(y)
        side = J[1] - J[7]; side -= y * (side @ y); side /= np.linalg.norm(side)
        n = np.cross(side, y); self.frame = np.stack([side, y, n])          # rows: hand X, Y, Z in MANO coordinates
        # flexion axes from the flat template: positive rotation curls each bone towards the palm
        self.axes = {}
        across = -side                                                       # thumb: towards the palm and the fingers
        for c, js in CHAINS.items():
            for k, j in enumerate(js):
                child = js[k + 1] if k + 1 < len(js) else None
                b = (J[child] - J[j]) if child is not None else (J[j] - J[js[k - 1]])
                b /= np.linalg.norm(b)
                to = n if c != 'thumb' or j == 13 else (.75 * n + .65 * across)   # thumb base lifts, then the thumb wraps
                a = np.cross(b, to); self.axes[j] = a / np.linalg.norm(a)
        self.digit_of = {j: c for c, js in CHAINS.items() for j in js}
        hz = (self.v0 - J[0]) @ self.frame.T
        palm = (self.owner == 0)
        self.PALM = dict(t=float(hz[palm, 2].max()), l=float(((J[4] - J[0]) @ self.frame.T)[1]))
        self.H = Matrix([list(r) + [0] for r in self.frame.tolist()] + [[0, 0, 0, 1]]) @ Matrix.Translation(Vector((-J[0]).tolist()))
        me = bpy.data.meshes.new('mano'); me.from_pydata(self.v0.tolist(), [], self.f.tolist())
        for p in me.polygons: p.use_smooth = True
        m = bpy.data.materials.get('skin') or bpy.data.materials.new('skin'); m.diffuse_color = (*SKIN_RGB, 1.); m.roughness = .55
        me.materials.append(m)
        self.root = bpy.data.objects.new('hand_root', None); bpy.context.scene.collection.objects.link(self.root)
        self.obj = bpy.data.objects.new('mano', me); bpy.context.scene.collection.objects.link(self.obj)
        self.obj.parent = self.root; self.obj.matrix_parent_inverse = Matrix.Identity(4); self.obj.matrix_basis = self.H
        sub = self.obj.modifiers.new('smooth', 'SUBSURF'); sub.levels = sub.render_levels = 2
        self.fr = {}; self.set({})

    # ------------------------------------------------------------------ posing
    def verts(self, fr):
        """fr: {joint index: 0..1 of its flexion range}. Returns posed vertices (MANO coordinates)."""
        R = np.stack([np.eye(3)] * 16)
        for j in range(1, 16):
            lim = FLEX['thumb' if self.digit_of[j] == 'thumb' else 'finger'][(j - 1) % 3]
            R[j] = _rodrigues(self.axes[j] * math.radians(lim) * fr.get(j, 0.))
        v = self.v0 + self.P @ (R[1:] - np.eye(3)).reshape(-1)
        G = np.zeros((16, 4, 4))
        for j in range(16):
            L = np.eye(4); L[:3, :3] = R[j]; L[:3, 3] = self.J[j] - (self.J[self.parent[j]] if j else 0)
            G[j] = L if j == 0 else G[self.parent[j]] @ L
        for j in range(16): G[j, :3, 3] -= G[j, :3, :3] @ self.J[j]
        T = np.einsum('vj,jab->vab', self.W, G)
        return np.einsum('vab,vb->va', T[:, :3, :3], v) + T[:, :3, 3]

    def set(self, fr):
        self.fr = dict(fr); self.V = self.verts(fr)
        self.obj.data.vertices.foreach_set('co', self.V.reshape(-1)); self.obj.data.update(); bpy.context.view_layer.update()

    def digit_tree(self, c, V=None):
        V = self.V if V is None else V; M = self.obj.matrix_world
        js = set(CHAINS[c]); keep = [p for p in self.f if sum(self.owner[i] in js for i in p) >= 2]
        idx = sorted({i for p in keep for i in p}); remap = {i: k for k, i in enumerate(idx)}
        pts = [M @ Vector(V[i].tolist()) for i in idx]
        return BVHTree.FromPolygons(pts, [[remap[i] for i in p] for p in keep])

    def close(self, tool_tree, steps=60):
        """Each digit closes until it touches the tool, then its outer joints curl on until they touch too."""
        fr = {}
        for c in ('index', 'middle', 'ring', 'pinky', 'thumb'):
            js = CHAINS[c]
            for stage in ((js, js[1:], js[2:]) if c != 'thumb' else (js[:1], js[1:], js[2:])):
                for _ in range(steps):
                    if all(fr.get(j, 0.) >= 1. for j in stage): break
                    trial = {**fr, **{j: min(1., fr.get(j, 0.) + 1. / steps) for j in stage}}
                    if self.digit_tree(c, self.verts(trial)).overlap(tool_tree): break
                    fr = trial
        self.set(fr); return fr

    def tree(self):
        """BVH of the whole posed hand (world space)."""
        M = self.obj.matrix_world
        return BVHTree.FromPolygons([M @ Vector(v.tolist()) for v in self.V], self.f.tolist())


def seat(hand, tool, place, bvh, step=.001):
    """Lift the tool along the palm normal (hand +Z) until it clears the open hand; place(dz) -> tool world matrix."""
    hand.set({}); dz = 0.
    while dz < .03:
        tool.matrix_world = place(dz); bpy.context.view_layer.update()
        if not hand.tree().overlap(bvh(tool)): break
        dz += step
    return dz
