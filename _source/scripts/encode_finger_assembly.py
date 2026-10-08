"""Encode and verify the seven-second full-boundary finger assembly sequence."""
import sys
sys.dont_write_bytecode=True
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/finger-assembly'

def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    frames=manifest['frames'];order=manifest['order']
    assert len(frames)==184 and len(order)==210
    assert [f['index'] for f in frames]==list(range(184))
    for name,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest
    orbit=np.unwrap(np.array([f['locations'][0] for f in frames[:91]])*2*np.pi)
    assert abs((orbit[-1]-orbit[0])/(2*np.pi)-1)<1e-6
    settled={identity:frames[index]['bases_mm'][frames[index]['identities'].index(identity)]
             for identity,index in enumerate((90,121,152,183))}
    for f in frames:
        assert len(f['identities'])==f['stage']
        assert f['palette_identity']==['t','f3','f2','f1']
        assert all(.025<v<.975 for v in f['bounds'])
        if f['minimum_spacing_mm'] is not None:assert f['minimum_spacing_mm']>=40.5
        for identity,base in zip(f['identities'],f['bases_mm']):
            if identity<f['stage']-1:assert np.allclose(base,settled[identity],atol=1e-3)
    final=frames[-1]
    for identity,normal in zip(final['identities'],final['normals']):
        assert np.allclose(normal,[1,0] if identity==0 else [0,1],atol=1e-5)
    fonts=[ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n) for n in (32,26,20)]
    @lru_cache(maxsize=4)
    def image(index):
        f=frames[index]
        raw=Image.open(OUT/'frames'/f'{index:04d}.png').convert('RGB')
        assert raw.size==(1200,1000)
        assert all(min(raw.getpixel(p))>=250 for p in [(0,0),(1199,0),(0,999),(1199,999)])
        result=Image.new('RGB',(1200,1200),'white');result.paste(raw,(0,112))
        draw=ImageDraw.Draw(result)
        draw.text((48,24),'Finger placement',font=fonts[0],fill='#18202a')
        label='1 finger | Full palm-boundary circuit' if f['stage']==1 else f'{f["stage"]} fingers | Adding finger {f["stage"]}'
        if index==183:label='4 fingers | Three on top, one on the right'
        draw.text((48,68),label,font=fonts[1],fill='#4a515b')
        draw.line((48,108,1152,108),fill='#e7e9eb',width=1)
        draw.text((48,1125),'Boundary locations: '+', '.join(f'{t:.3f}' for t in f['locations']),font=fonts[2],fill='#626a74')
        draw.text((48,1155),'Finger-only mode | Palm size: 130 mm',font=fonts[2],fill='#626a74')
        bar=1200/.54*.02
        draw.line((1040,1142,1040+bar,1142),fill='#626a74',width=2)
        for x in (1040,1040+bar):draw.line((x,1137,x,1147),fill='#626a74',width=2)
        draw.text((1038,1154),'20 mm',font=fonts[2],fill='#626a74')
        return result
    video=OUT/'finger-assembly-7s.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1200x1200','-r','30','-i','-','-an','-c:v','libx264','-preset','medium',
        '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    try:
        for index in order:process.stdin.write(image(index).tobytes())
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('FFmpeg encoding failed.')
    finally:
        if process.poll() is None:process.kill()
    image(183).save(OUT/'poster.png')
    stages=Image.new('RGB',(1200,1200),'white')
    for n,index in enumerate((45,121,152,183)):
        stages.paste(image(index).resize((600,600),Image.Resampling.LANCZOS),((n%2)*600,(n//2)*600))
    stages.save(OUT/'stages.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
    stream=probe['streams'][0]
    assert stream['r_frame_rate']=='30/1' and int(stream['nb_frames'])==210
    assert (stream['width'],stream['height'])==(1200,1200)
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    assert abs(float(probe['format']['duration'])-7)<.01
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    report=dict(file=video.name,duration_seconds=7,fps=30,frames=210,native_renders=184,
                full_boundary_circuits=1,finger_counts=[1,2,3,4],final_layout='Three top, one right',
                settled_fingers_remain_fixed=True,minimum_base_spacing_mm=min(f['minimum_spacing_mm'] for f in frames if f['minimum_spacing_mm'] is not None),
                source_hashes='passed',framing='passed',full_decode='passed',bytes=video.stat().st_size)
    (OUT/'video-checks.json').write_text(json.dumps(report,indent=2))
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Finger placement sequence</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#18202a;background:white}h1{font-size:28px}video{display:block;width:100%;max-height:78vh}a{color:#245889}</style><h1>Finger placement</h1><p>The first finger makes a full circuit and settles on the right. Three additional fingers arrive individually and settle along the top.</p><video controls preload="metadata" poster="poster.png" src="finger-assembly-7s.mp4"></video><p><a href="finger-assembly-7s.mp4" download>Download the 7-second video</a></p><p>Original generator in finger-only mode, original material palette, fixed front camera, white background and approved cavity lighting. Colors stay associated with individual fingers.</p></html>''',encoding='utf-8')
    (OUT/'render-notes.txt').write_text('''Finger placement sequence

Native generation_v2 finger_only mode, zero thumbs. All four digits use the
original 2-1-1 finger structure. Palm size 130 mm, aspect ratio 1.55.
The first finger completes one full boundary circuit and stays on the lower
right. Fingers 2, 3 and 4 arrive from the upper right and settle at the left,
center and right of the top edge, respectively. The native location and
spacing checks run for every generated configuration; finger positions are
not changed by the native spacing correction.

Original materials are assigned by persistent identity: t, f3, f2, f1.
No material colors or shader nodes are changed. The first finger uses the
original thumb material because it occupies the final thumb position.

Workbench studio mylight.sl; Both cavity; World ridge/valley 2.5/2.5,
Screen ridge/valley 2/2. Fixed front orthographic camera, scale 0.54 m.
184 native renders; 210 output frames at 30 fps, duration 7 seconds.
No geometry interpolation or simulated grasp motion. The original generator
rebuilds the palm attachment geometry at every sample.

Reproduce: render_finger_assembly.py --plan inside Blender, then run
run_finger_assembly.py and encode_finger_assembly.py with normal Python.
All output remains inside website_assets; the research webpage is unchanged.
''',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
