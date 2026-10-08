"""Export the full palm overview at the requested duration and update its review page."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1] / 'preview/palm-parameters'
source = ROOT / 'videos/palm-parameters-overview.mp4'
parser = argparse.ArgumentParser()
parser.add_argument('--seconds', type=int, default=7)
args = parser.parse_args()
if args.seconds <= 0:
    raise ValueError('Duration must be positive.')
duration, fps = args.seconds, 30
target = ROOT / f'videos/palm-parameters-overview-{duration}s.mp4'

def probe(path):
    return json.loads(subprocess.check_output([
        'ffprobe', '-v', 'error', '-show_streams', '-show_format',
        '-of', 'json', str(path)], text=True))

original_duration = float(probe(source)['format']['duration'])
subprocess.run([
    'ffmpeg', '-y', '-loglevel', 'error', '-i', str(source),
    '-vf', f'setpts=(PTS-STARTPTS)*{duration/original_duration:.12f},fps={fps},tpad=stop_mode=clone:stop_duration=1',
    '-frames:v', str(duration*fps), '-an', '-c:v', 'libx264',
    '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
    '-movflags', '+faststart', str(target)], check=True)
result = probe(target)
stream = result['streams'][0]
assert abs(float(result['format']['duration']) - duration) < .01
assert int(stream['nb_frames']) == duration*fps
assert stream['r_frame_rate'] == f'{fps}/1'
assert (stream['width'], stream['height']) == (1200, 1200)
subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(target), '-f', 'null', '-'], check=True)
report = dict(file=target.name, seconds=duration, fps=fps, frames=duration*fps,
              speed_multiplier=original_duration/duration, source=source.name,
              bytes=target.stat().st_size, full_decode='passed')
(ROOT/f'video-{duration}s-checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
page = ROOT/'index.html'
content = page.read_text(encoding='utf-8')
content = re.sub(r'videos/palm-parameters-overview(?:-\d+s)?\.mp4', 'videos/'+target.name, content)
content = re.sub(r'Download combined video(?: \(\d+ seconds\))?</a>', f'Download combined video ({duration} seconds)</a>', content)
page.write_text(content, encoding='utf-8')
print(json.dumps(report, indent=2))
