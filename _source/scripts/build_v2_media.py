"""Build web media for the v2 page (website_assets/v2/). Sources are read only; nothing is removed.

Run from the repository root:  python -B website_assets/scripts/build_v2_media.py
Requires FFmpeg on PATH.
"""
import json
import subprocess
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
REPO = SITE.parent
OUT = SITE / 'v2/media'
DEMOS = REPO / 'iros_paper_codebase/optimization/grasp_evaluation/data'

# Human demonstrations with 6D object-pose tracking (task trajectories used by the grasp evaluation).
DEMO_TASKS = ['stir', 'hammer', 'knife', 'saw', 'scoup']
DEMO_SECONDS = 12

# Grasp search renders from the original project page.
CLIPS = [
    ('grasp-pose-sampling', SITE / 'assets/graspsample.mp4'),
    ('joint-space-search', SITE / 'assets/grasp.mp4'),
]

# Crops of the paper figures: (name, source, w, h, x, y) in source pixels.
SIM_W, SIM_H = 16560, 7065
CROPS = [
    ('framework-sampling', 'figs/Framework_v2.jpg', 1550, 1640, 0, 650),
    ('framework-wrench', 'figs/Framework_v2.jpg', 980, 620, 372, 3140),
    ('framework-demos', 'figs/Framework_v2.jpg', 1000, 1000, 1200, 2050),
    ('design-high', 'figs/SimResults_v2.jpg', int(SIM_W * .215), int(SIM_H * .265), int(SIM_W * .783), int(SIM_H * .122)),
    ('design-mid', 'figs/SimResults_v2.jpg', int(SIM_W * .215), int(SIM_H * .275), int(SIM_W * .783), int(SIM_H * .400)),
    ('design-low', 'figs/SimResults_v2.jpg', int(SIM_W * .215), int(SIM_H * .285), int(SIM_W * .783), int(SIM_H * .685)),
    ('teaser-sim', 'figs/Teaser.jpg', int(6009 * .38), int(4542 * .88), 0, int(4542 * .12)),
    ('teaser-real', 'figs/Teaser.jpg', int(6009 * .375), int(4542 * .88), int(6009 * .38), int(4542 * .12)),
    ('teaser-task', 'figs/Teaser.jpg', int(6009 * .245), int(4542 * .88), int(6009 * .755), int(4542 * .12)),
]


def ffmpeg(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *map(str, args)], check=True)


def duration(path):
    out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(path)],
                         check=True, capture_output=True, text=True).stdout
    return float(out.strip())


def encode(src, dst, vf, crf=26):
    ffmpeg('-i', src, '-vf', vf, '-an', '-c:v', 'libx264', '-crf', crf, '-preset', 'slow', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', dst)


def poster(video, dst, at):
    ffmpeg('-ss', at, '-i', video, '-frames:v', '1', '-quality', '84', dst)


# Framework-slide animations from the IROS presentation (assets/*.mp4 are the slide's GIFs), cropped to the
# union of the hand's extent over all frames (measured on the white background) plus a margin.
PIPELINE = {'general': (392, 92, 1528, 951), 'surface': (741, 159, 1476, 871), 'fingers': (646, 158, 1480, 879),
            'graspsample': (535, 120, 1482, 1080), 'sidejoints': (595, 216, 1385, 964), 'grasp': (177, 132, 1356, 989)}


def pipeline():
    out = OUT / 'pipeline'; out.mkdir(parents=True, exist_ok=True)
    for name, (l, t, r, b) in PIPELINE.items():
        l, t, r, b = max(0, l - 30), max(0, t - 30), min(1920, r + 30), min(1080, b + 30)
        w, h = (r - l) // 2 * 2, (b - t) // 2 * 2
        dst = out / f'{name}.mp4'
        encode(SITE / f'assets/{name}.mp4', dst, f'crop={w}:{h}:{l}:{t},scale=520:-2,setsar=1', crf=25)
        poster(dst, out / f'{name}.webp', duration(dst) * .5)
    print(f'Wrote {len(PIPELINE)} pipeline clips to {out}')


def main():
    if '--pipeline' in sys.argv:
        pipeline(); return
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for task in DEMO_TASKS:
        src = DEMOS / task / 'track_vis.mp4'
        speed = max(1.0, duration(src) / DEMO_SECONDS)
        dst = OUT / f'demo-{task}.mp4'
        encode(src, dst, f'setpts=PTS/{speed:.4f},fps=30,scale=640:480,setsar=1', crf=27)
        poster(dst, OUT / f'demo-{task}.webp', DEMO_SECONDS * .6)
        records.append({'id': f'demo-{task}', 'source': str(src.relative_to(REPO)).replace('\\', '/'), 'speed': round(speed, 3)})
    for name, src in CLIPS:
        dst = OUT / f'{name}.mp4'
        encode(src, dst, 'scale=1280:-2,setsar=1', crf=24)
        poster(dst, OUT / f'{name}.webp', duration(dst) * .5)
        records.append({'id': name, 'source': str(src.relative_to(REPO)).replace('\\', '/')})
    for name, rel, w, h, x, y in CROPS:
        ffmpeg('-i', SITE / rel, '-vf', f"crop={w}:{h}:{x}:{y},scale='min(1100,iw)':-2", '-quality', '86', OUT / f'{name}.webp')
        records.append({'id': name, 'source': rel, 'crop': [w, h, x, y]})
    (OUT / 'manifest.json').write_text(json.dumps(records, indent=2))
    print(f'Wrote {len(records)} items to {OUT}')


if __name__ == '__main__':
    main()
