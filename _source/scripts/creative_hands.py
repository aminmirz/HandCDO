"""Hand designs for the v2 page's 3D stage: compact palms, fingers placed anywhere on the closed outline,
thumbless and non-anthropomorphic layouts. Plain Python (no Blender): builds the native generation_v2
PalmConfig / FingerConfig / Hand objects. export_creative_hands.py runs the same builder inside Blender.

Outline location t runs counter-clockwise from the middle of the right side: top 0.25, left 0.5, bottom 0.75.
Fingers are listed with (t, code, added link length mm, angle deg).
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parent / 'generation_v2'
for p in (ROOT / '.runtime/python', GEN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import math
import numpy as np

STD = '2-1-1'          # approved finger code
THUMB = '1--22'        # approved thumb code

# id: label, palm (size mm, sides, aspect, rotation), fingers [(t, code, link mm, angle)], thumb (t, code, link mm) or None
DESIGNS = {
    'compact': ('Compact 3+1', (110, 4, 1.45, 0),
                [(.155, STD, 0, 0), (.25, STD, 0, 0), (.345, STD, 0, 0)], (.955, THUMB, 0)),
    'duo': ('2 fingers + thumb', (96, 4, 1.2, 0),
            [(.19, STD, 2, 0), (.31, STD, 2, 0)], (.95, THUMB, 2)),
    'side-four': ('4 + thumb, side finger', (118, 4, 1.35, 0),
                  [(.16, STD, 0, 0), (.25, STD, 0, 0), (.34, STD, 0, 0), (.5, '2-1-12', 4, 0)], (.955, THUMB, 0)),
    'radial-four': ('Radial 4, no thumb', (100, 4, 1.0, 0),
                    [(0., '2-1-11', 4, 0), (.25, '2-1-11', 4, 0), (.5, '2-1-11', 4, 0), (.75, '2-1-11', 4, 0)], None),
    'tri-claw': ('Tri-claw, no thumb', (92, 6, 1.0, 0),
                 [(.083, '0--111', 6, 0), (.417, '0--111', 6, 0), (.75, '0--111', 6, 0)], None),
    'opposed': ('Top + bottom fingers', (104, 4, 1.25, 0),
                [(.19, STD, 2, 0), (.31, STD, 2, 0), (.69, STD, 2, 0), (.81, STD, 2, 0)], None),
    'pinch-plus': ('3 top, 2 bottom', (118, 4, 1.4, 0),
                   [(.16, STD, 0, 0), (.25, STD, 0, 0), (.34, STD, 0, 0), (.69, '2-1-12', 4, 0), (.81, '2-1-12', 4, 0)], None),
    'star-five': ('Pentagon star, no thumb', (118, 5, 1.0, 0),
                  [(.05, '1-1-11', 6, 0), (.25, '1-1-11', 6, 0), (.45, '1-1-11', 6, 0), (.65, '1-1-11', 6, 0), (.85, '1-1-11', 6, 0)], None),
    'long-reach': ('Large palm, long digits', (150, 4, 1.5, 0),
                   [(.155, '2-11-11', 10, 6), (.25, '2-11-11', 10, 0), (.345, '2-11-11', 10, -6)], (.955, '1--2222', 10)),
}


def build(design, root_dir):
    """Return the native Hand for a design id."""
    from Config import PalmConfig, FingerConfig, HandConfig
    from HandClass import Hand
    label, (size, sides, aspect, rot), fingers, thumb = DESIGNS[design]
    cfg = PalmConfig(data='fixed', detailed_viz=True)   # Detailed Viz: mounts, screw holes, cable cut-outs
    cfg.palm_size_mm = float(size); cfg.outline_sides = sides; cfg.outline_aspect_ratio = float(aspect)
    cfg.outline_rotation_deg = float(rot); cfg.generation_mode = 'standard'
    n = len(fingers); cfg.finger_number = n; cfg.thumb_number = 1 if thumb else 0
    cfg.bumps_number = 0; cfg.bump_max_height_intensity_mm = 0.; cfg.pad_resolution_level = 4
    cfg.initialize_outline()
    curve = np.vstack([cfg.outline_points, cfg.outline_points[0]])
    cfg.finger_location_list = [f[0] for f in fingers]
    cfg.finger_points = np.array([cfg._curve_point_at(curve, f[0]) for f in fingers]).reshape(n, 2)
    cfg.finger_angle_deg_list = [float(f[3]) for f in fingers]
    cfg.finger_base_normal_offset_mm_list = [0.] * n; cfg.finger_base_side_offset_mm_list = [0.] * n
    if thumb:
        cfg.thumb_side_list = []; cfg.thumb_location_list = [thumb[0]]
        cfg.thumb_base_normal_offset_mm_list = [0.]; cfg.thumb_base_side_offset_mm_list = [0.]
        # align the thumb's base direction with the outward boundary normal (as the approved renders do)
        cfg.thumb_angle_deg_list = [-math.degrees(math.atan2(cfg.thumb_corner_seg1_mm, cfg.thumb_corner_seg2_mm))]
        cfg.thumb_points = np.array([cfg._curve_point_at(curve, thumb[0])])
    else:
        for name in ('thumb_side_list', 'thumb_location_list', 'thumb_base_normal_offset_mm_list',
                     'thumb_base_side_offset_mm_list', 'thumb_angle_deg_list'):
            setattr(cfg, name, [])
        cfg.thumb_points = np.empty((0, 2))
    finger_cfgs = []
    for i, (t, code, link, angle) in enumerate(fingers):
        fc = FingerConfig(type='finger', data='fixed', code=code, id=i)
        fc.link_added_length_mm_list = [float(link)] * len(fc.link_added_length_mm_list)
        finger_cfgs.append(fc)
    thumb_cfgs = []
    if thumb:
        tc = FingerConfig(type='thumb', data='fixed', code=thumb[1], id=0)
        tc.link_added_length_mm_list = [float(thumb[2])] * len(tc.link_added_length_mm_list)
        thumb_cfgs.append(tc)
    hc = HandConfig(); hc.collision_mesh = False
    return Hand(cfg, finger_cfgs, thumb_cfgs, hc, root_dir=str(root_dir)), cfg


def sketch(designs, out):
    """Top-view sketch of palm outline, bases and digit directions for quick layout review."""
    from PIL import Image, ImageDraw, ImageFont
    import tempfile
    W = 380; img = Image.new('RGB', (W * min(len(designs), 5), W * math.ceil(len(designs) / 5)), 'white')
    d = ImageDraw.Draw(img); font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 15)
    for k, design in enumerate(designs):
        ox, oy = (k % 5) * W + W / 2, (k // 5) * W + W / 2 + 10; s = 1.1
        P = lambda p: (ox + p[0] * s, oy - p[1] * s)
        try:
            hand, cfg = build(design, tempfile.mkdtemp())
            outline = np.asarray(cfg.outline_points)
            d.polygon([P(p) for p in outline], fill='#e8c9c9', outline='#8a1c22')
            colors = ['#2fc46a', '#e2c52e', '#3b82f6', '#9b59b6', '#16a085', '#e67e22']
            for i, (b, nv) in enumerate(zip(cfg.finger_bases, cfg.finger_bases_normal_vectors)):
                length = 95 + 10 * DESIGNS[design][2][i][2] / 4
                d.line([P(b), P(b + nv * length)], fill=colors[i % 6], width=9)
            for b, nv in zip(getattr(cfg, 'thumb_bases', []), getattr(cfg, 'thumb_bases_normal_vectors', [])):
                d.line([P(b), P(b + nv * 85)], fill='#ef4f2c', width=9)
            ext = outline.max(0) - outline.min(0)
            d.text((ox - W / 2 + 8, oy - W / 2 - 5), f'{design}  palm {ext[0]:.0f}x{ext[1]:.0f} mm', fill='black', font=font)
        except Exception as e:
            d.text((ox - W / 2 + 8, oy), f'{design}: {type(e).__name__} {e}'[:60], fill='red', font=font)
    img.save(out)


if __name__ == '__main__':
    sketch(list(DESIGNS), sys.argv[1] if len(sys.argv) > 1 else 'creative-sketch.png')
