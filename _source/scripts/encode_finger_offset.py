"""Encode and verify the approved four-finger normal-offset demonstration."""
import sys
sys.dont_write_bytecode=True
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/finger-offset'

def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    baseline=json.loads((ROOT/'preview/finger-assembly/frames/0183.json').read_text())
    frames=manifest['frames'];order=manifest['order']
    assert len(frames)==91 and len(order)==210
    assert [f['index'] for f in frames]==list(range(91))
    assert frames[0]['offset_mm']==0 and frames[-1]['offset_mm']==25
    for name,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest
    for f in frames:
        assert f['identities']==baseline['identities'] and f['locations']==baseline['locations']
        assert f['palette_identity']==baseline['palette_identity']
        assert all(.025<v<.975 for v in f['bounds'])
        assert np.allclose(f['normals'],baseline['normals'],atol=1e-6)
        for rank,identity in enumerate(f['identities']):
            delta=np.array(f['bases_mm'][rank])-np.array(baseline['bases_mm'][rank])
            expected=np.array(baseline['normals'][rank])*(0 if identity==0 else f['offset_mm'])
            assert np.allclose(delta,expected,atol=1e-3),(f['index'],identity,delta,expected)
    assert all(a['offset_mm']<=b['offset_mm'] for a,b in zip(frames,frames[1:]))
    fonts=[ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n) for n in (32,26,20)]
    @lru_cache(maxsize=4)
    def image(index):
        f=frames[index]
        raw=Image.open(OUT/'frames'/f'{index:04d}.png').convert('RGB')
        assert raw.size==(1200,1000)
        assert all(min(raw.getpixel(p))>=250 for p in [(0,0),(1199,0),(0,999),(1199,999)])
        result=Image.new('RGB',(1200,1200),'white');result.paste(raw,(0,112))
        draw=ImageDraw.Draw(result)
        draw.text((48,24),'Finger base offset',font=fonts[0],fill='#18202a')
        draw.text((48,68),f'Top finger normal offset: {f["offset_mm"]:.1f} mm',font=fonts[1],fill='#4a515b')
        draw.line((48,108,1152,108),fill='#e7e9eb',width=1)
        draw.text((48,1125),'Three top fingers move outward; the right-side finger stays fixed.',font=fonts[2],fill='#626a74')
        draw.text((48,1155),'Finger-only mode | Palm size: 130 mm | Fixed finger geometry',font=fonts[2],fill='#626a74')
        bar=1200/manifest['camera']['scale_m']*.02
        draw.line((1040,1142,1040+bar,1142),fill='#626a74',width=2)
        for x in (1040,1040+bar):draw.line((x,1137,x,1147),fill='#626a74',width=2)
        draw.text((1038,1154),'20 mm',font=fonts[2],fill='#626a74')
        return result
    assert ImageChops.difference(image(0).crop((0,112,1200,1112)),image(90).crop((0,112,1200,1112))).getbbox()
    video=OUT/'finger-offset-7s.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1200x1200','-r','30','-i','-','-an','-c:v','libx264','-preset','medium',
        '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    try:
        for index in order:process.stdin.write(image(index).tobytes())
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('FFmpeg encoding failed.')
    finally:
        if process.poll() is None:process.kill()
    image(0).save(OUT/'poster.png')
    comparison=Image.new('RGB',(1200,600),'white')
    for n,index in enumerate((0,90)):
        comparison.paste(image(index).resize((600,600),Image.Resampling.LANCZOS),(n*600,0))
    comparison.save(OUT/'endpoints.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
    stream=probe['streams'][0]
    assert stream['r_frame_rate']=='30/1' and int(stream['nb_frames'])==210
    assert (stream['width'],stream['height'])==(1200,1200)
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    assert abs(float(probe['format']['duration'])-7)<.01
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    report=dict(file=video.name,seconds=7,frames=210,fps=30,native_renders=91,offset_range_mm=[0,25],
                approved_baseline='passed',right_side_finger_fixed=True,top_finger_displacements='passed',
                source_hashes='passed',framing='passed',full_decode='passed',bytes=video.stat().st_size)
    (OUT/'video-checks.json').write_text(json.dumps(report,indent=2))
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Finger base offset</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#18202a;background:white}h1{font-size:28px}video{display:block;width:100%;max-height:78vh}a{color:#245889}</style><h1>Finger base offset</h1><p>The three top finger bases move outward from 0 to 25 mm and return. The right-side finger stays fixed.</p><video controls loop preload="metadata" poster="poster.png" src="finger-offset-7s.mp4"></video><p><a href="finger-offset-7s.mp4" download>Download the 7-second video</a></p><p>Approved four-finger layout, original generator geometry and materials, fixed front camera and cavity lighting. Only the top fingers' normal-offset parameters change; the generator rebuilds their palm attachments.</p></html>''',encoding='utf-8')
    (OUT/'render-notes.txt').write_text('''Finger base offset

Uses the approved final four-finger layout from preview/finger-assembly.
Original generation_v2 finger_only mode; palm size 130 mm, aspect ratio 1.55.
Three upward-pointing finger bases sweep along their normals from 0 to 25 mm
and return. The horizontal right-side finger has zero offset throughout.
Finger locations, joint structure, link dimensions and material identities
remain fixed. The native generator rebuilds each frame, including the palm
attachment geometry. Workbench mylight.sl, Both cavity, World 2.5/2.5,
Screen 2/2; white background and the approved front camera at scale 0.54 m.

91 unique native renders, 210 encoded frames at 30 fps, 7-second loop.
Reproduce with plan_finger_offset.py, then run_finger_assembly.py --output
website_assets/preview/finger-offset, then encode_finger_offset.py.
All outputs are inside website_assets. The research webpage is unchanged.
''',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
