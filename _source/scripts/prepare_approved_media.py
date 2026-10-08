"""Create web copies of the eight approved animations; preserve every source."""
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'site/media/approved'
STUDIES = [
    ('palm-geometry', 'palm-parameters/videos/palm-parameters-overview-7s.mp4', 7, 3.3),
    ('digit-placement', 'thumb-assembly/thumb-and-fingers-7s.mp4', 7, 6.8),
    ('finger-offsets', 'individual-finger-offsets/individual-finger-offsets-4s.mp4', 4, 1.4),
    ('base-angles', 'combined-base-angles/combined-base-angles-3s.mp4', 3, 1.5),
    ('link-lengths', 'independent-link-lengths/independent-link-lengths-3s.mp4', 3, 1.4),
    ('joint-configurations', 'simultaneous-kinematics/simultaneous-kinematics-5s.mp4', 5, 2.2),
    ('contact-surfaces', 'contact-surfaces-front/contact-surfaces-front-5s.mp4', 5, 2.5),
    ('fingertip-geometry', 'fingertip-geometry/fingertip-geometry-5s.mp4', 5, 2.4),
]

def run(*args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *map(str, args)], check=True)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records = []
    for slug, relative, duration, poster_time in STUDIES:
        source = ROOT / 'preview' / relative
        folder = source.parent
        manifests = list((folder.parent / 'frames').glob('*/manifest.json')) if slug == 'palm-geometry' else [folder / 'manifest.json']
        bounds = [f['bounds'] for path in manifests for f in json.loads(path.read_text())['frames']]
        # Raw render occupies 1200 x 1000 pixels, below the 112px title band.
        left = max(0, math.floor(min(b[0] for b in bounds)*1200)-44)
        right = min(1200, math.ceil(max(b[2] for b in bounds)*1200)+44)
        top = max(0, math.floor((1-max(b[3] for b in bounds))*1000)-44)
        bottom = min(1000, math.ceil((1-min(b[1] for b in bounds))*1000)+44)
        left, top = left//2*2, top//2*2
        width, height = math.ceil((right-left)/2)*2, math.ceil((bottom-top)/2)*2
        vf = f'crop={width}:{height}:{left}:{112+top},scale=800:600:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=800:600:(ow-iw)/2:(oh-ih)/2:white,setsar=1'
        run('-i', source, '-vf', vf, '-an', '-c:v', 'libx264', '-crf', '19', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', OUT/f'{slug}.mp4')
        run('-ss', poster_time, '-i', OUT/f'{slug}.mp4', '-frames:v', '1', '-quality', '88', OUT/f'{slug}.webp')
        records.append(dict(id=slug, source=f'preview/{relative}', duration=duration, width=800, height=600, crop=[left,112+top,width,height], poster_time=poster_time))
        print(slug, 'ready', flush=True)
    (OUT/'manifest.json').write_text(json.dumps(records, indent=2)+'\n')

if __name__ == '__main__':
    main()
