"""Render (Blender) and encode the co-design loop clips: task, close, wrench -> website_assets/v2/media/clips/.

  python -B website_assets/scripts/build_framework_clips.py [--clips task close wrench] [--encode-only]
"""
import sys
sys.dont_write_bytecode = True
import argparse, json, os, subprocess
from pathlib import Path
from PIL import Image, ImageChops, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'preview/framework-clips'
OUT = ROOT / 'v2/media/clips'
BLENDER = os.environ.get('HANDCDO_BLENDER', 'C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')
FPS, W, H = 30, 960, 720


def render(clip):
    with (SRC / f'{clip}.log').open('w', encoding='utf-8') as log:
        subprocess.run([BLENDER, '-b', '--factory-startup', '--python-exit-code', '1', '--python',
                        str(ROOT / 'scripts/render_framework_clips.py'), '--', '--clip', clip], stdout=log, stderr=subprocess.STDOUT, check=True)


def content_box(frames):
    box = None
    for p in frames[::4]:
        im = Image.open(p).convert('RGB'); b = ImageChops.difference(im, Image.new('RGB', im.size, 'white')).convert('L').point(lambda v: 255 if v > 12 else 0).getbbox()
        if b: box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
    l, t, r, b = box; pad = 40; l, t, r, b = l - pad, t - pad - 30, r + pad, b + pad    # room for the label
    w, h = r - l, b - t
    if w / h < W / H: l -= (h * W / H - w) / 2; w = h * W / H
    else: t -= (w * H / W - h) / 2; h = w * H / W
    return tuple(int(round(v)) for v in (l, t, l + w, t + h))


def encode(clip):
    labels = json.loads((SRC / clip / 'labels.json').read_text())
    frames = sorted((SRC / clip).glob('0*.png'))[:len(labels)]                 # frames left from a longer earlier render are ignored
    box = content_box(frames); font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 26)
    OUT.mkdir(parents=True, exist_ok=True); video = OUT / f'{clip}.mp4'
    proc = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                             '-an', '-c:v', 'libx264', '-preset', 'slow', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    poster = None
    for i, (p, lab) in enumerate(zip(frames, labels)):
        im = Image.open(p).convert('RGB'); canvas = Image.new('RGB', (box[2] - box[0], box[3] - box[1]), 'white')
        canvas.paste(im, (-box[0], -box[1])); canvas = canvas.resize((W, H), Image.Resampling.LANCZOS)
        ImageDraw.Draw(canvas).text((28, 22), lab, font=font, fill='#4a515b')
        proc.stdin.write(canvas.tobytes())
        if i == int(len(frames) * .7): poster = canvas
    proc.stdin.close(); assert proc.wait() == 0
    poster.save(OUT / f'{clip}.webp', quality=86)
    print(f'{clip}: {len(frames)} frames, {len(frames) / FPS:.1f}s, crop {box}', flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--clips', nargs='+', default=['task', 'close', 'wrench']); ap.add_argument('--encode-only', action='store_true')
    a = ap.parse_args()
    for c in a.clips:
        if not a.encode_only: render(c)
        encode(c)


if __name__ == '__main__':
    main()
