"""Encode the native finger-location sweep as a verified 7-second loop."""
import sys
sys.dont_write_bytecode=True
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageChops

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/finger-location'
SIZE=(1200,1200)
FPS=30

def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    frames=manifest['frames']
    assert [f['index'] for f in frames]==list(range(91))
    for name,digest in manifest['source_hashes'].items():
        assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest,name
    gaps=[f['spacing_mm'] for f in frames]
    assert min(gaps)>40.5 and max(gaps)-min(gaps)>19
    assert all(a<=b+1e-5 for a,b in zip(gaps,gaps[1:]))
    for f in frames:
        assert np.allclose(f['thumb_bases_mm'],frames[0]['thumb_bases_mm'],atol=1e-6)
        assert np.allclose(f['bases_mm'][1],frames[0]['bases_mm'][1],atol=1e-6)
        assert all(.025<v<.975 for v in f['bounds'])
    fonts=[ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n) for n in (32,26,20)]
    images=[]
    for f in frames:
        raw=Image.open(OUT/'frames'/f'{f["index"]:04d}.png').convert('RGB')
        assert raw.size==(1200,1000)
        assert all(min(raw.getpixel(p))>=250 for p in [(0,0),(1199,0),(0,999),(1199,999)])
        image=Image.new('RGB',SIZE,'white');image.paste(raw,(0,112))
        draw=ImageDraw.Draw(image)
        draw.text((48,24),'Finger location',font=fonts[0],fill='#18202a')
        draw.text((48,68),f'Adjacent base spacing: {f["spacing_mm"]:.1f} mm',font=fonts[1],fill='#4a515b')
        draw.line((48,108,1152,108),fill='#e7e9eb',width=1)
        draw.text((48,1125),'Boundary locations: '+', '.join(f'{v:.3f}' for v in f['locations']),font=fonts[2],fill='#626a74')
        draw.text((48,1155),'Palm size: 130 mm | Fixed finger dimensions and thumb pose',font=fonts[2],fill='#626a74')
        bar=1200/manifest['camera']['scale_m']*.020
        draw.line((1040,1142,1040+bar,1142),fill='#626a74',width=2)
        for x in (1040,1040+bar):draw.line((x,1137,x,1147),fill='#626a74',width=2)
        draw.text((1040,1154),'20 mm',font=fonts[2],fill='#626a74')
        images.append(image)
    assert ImageChops.difference(images[0].crop((0,112,1200,1112)),images[-1].crop((0,112,1200,1112))).getbbox()
    order=[0]*15+list(range(91))+[90]*15+list(range(89,0,-1))
    assert len(order)==210
    path=OUT/'finger-location-7s.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1200x1200','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium',
        '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    try:
        for i in order:process.stdin.write(images[i].tobytes())
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('FFmpeg encoding failed.')
    finally:
        if process.poll() is None:process.kill()
    images[0].save(OUT/'poster.png')
    comparison=Image.new('RGB',(1200,600),'white')
    comparison.paste(images[0].resize((600,600),Image.Resampling.LANCZOS),(0,0))
    comparison.paste(images[-1].resize((600,600),Image.Resampling.LANCZOS),(600,0))
    comparison.save(OUT/'endpoints.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],text=True))
    stream=probe['streams'][0]
    assert int(stream['nb_frames'])==210 and stream['r_frame_rate']=='30/1'
    assert (stream['width'],stream['height'])==SIZE and stream['codec_name']=='h264'
    assert abs(float(probe['format']['duration'])-7)<.01
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True)
    report=dict(file=path.name,seconds=7,frames=210,fps=30,unique_native_renders=91,
                spacing_range_mm=[min(gaps),max(gaps)],thumb_fixed=True,center_finger_fixed=True,
                framing='passed',white_background='passed',source_hashes='passed',full_decode='passed',bytes=path.stat().st_size)
    (OUT/'video-checks.json').write_text(json.dumps(report,indent=2))
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Finger location video</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#18202a;background:white}h1{font-size:28px}video{display:block;width:100%;max-height:78vh}a{color:#245889}</style><h1>Finger location</h1><p>Three fingers on a 130 mm palm. The outer finger bases move apart and return; the center finger and horizontal thumb remain fixed.</p><video controls loop preload="metadata" poster="poster.png" src="finger-location-7s.mp4"></video><p><a href="finger-location-7s.mp4" download>Download the 7-second video</a></p><p>Original generator geometry and materials, approved front view and cavity lighting. The generator rebuilds the attachment edges as the finger locations change.</p></html>''',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
