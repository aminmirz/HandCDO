"""Encode and validate the approved-shading native palm studies for review."""
import sys
sys.dont_write_bytecode=True
import hashlib
import html
import json
from pathlib import Path
import subprocess
from PIL import Image, ImageDraw, ImageFont, ImageChops

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/palm-parameters'
VIDEOS=OUT/'videos'
ORDER=['shape','size','aspect','rotation','smoothing','gallery']
FPS=24
SIZE=(1200,1200)
FONT='C:/Windows/Fonts/arial.ttf'
TITLE=ImageFont.truetype(FONT,32)
LABEL=ImageFont.truetype(FONT,26)
SMALL=ImageFont.truetype(FONT,20)

def frame_image(manifest,frame):
    image=Image.open(OUT/'frames'/manifest['study']/f'{frame["index"]:04d}.png').convert('RGB')
    assert image.size==(1200,1000)
    assert all(min(image.getpixel(p))>=250 for p in [(0,0),(1199,0),(0,999),(1199,999)])
    canvas=Image.new('RGB',SIZE,'white');canvas.paste(image,(0,112))
    draw=ImageDraw.Draw(canvas)
    draw.text((48,24),manifest['title'],font=TITLE,fill='#18202a')
    draw.text((48,68),frame['label'],font=LABEL,fill='#4a515b')
    draw.line((48,108,1152,108),fill='#e7e9eb',width=1)
    chapter=ORDER.index(manifest['study'])+1
    draw.text((1085,30),f'{chapter:02d} / 06',font=SMALL,fill='#626a74')
    note={
        'shape':'Width 130 mm  |  Aspect ratio 1.55  |  Rotation 0 degrees',
        'size':'4 sides  |  Aspect ratio 1.55  |  Rotation 0 degrees',
        'aspect':'4 sides  |  Width 130 mm  |  Rotation 0 degrees',
        'rotation':'4 sides  |  X width 130 mm (normalized by generator)',
        'smoothing':'4 sides  |  Width 130 mm  |  Outline resolution 10 mm',
        'gallery':'Combined changes to outline shape, size, aspect ratio and rotation'
    }[manifest['study']]
    draw.text((48,1144),note,font=SMALL,fill='#626a74')
    bar=1200/manifest.get('camera_scale_m',.28)*.020
    draw.line((1040,1142,1040+bar,1142),fill='#626a74',width=2)
    for x in (1040,1040+bar):draw.line((x,1137,x,1147),fill='#626a74',width=2)
    draw.text((1053,1154),'20 mm',font=SMALL,fill='#626a74')
    return canvas

def encode(path,frames):
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
       '-s','1200x1200','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','medium',
       '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    try:
        for frame in frames:process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('FFmpeg encoding failed.')
    finally:
        if process.poll() is None:process.kill()

def main():
    VIDEOS.mkdir(parents=True,exist_ok=True)
    chapters=[];posters=[]
    for study in ORDER:
        manifest=json.loads((OUT/'frames'/study/'manifest.json').read_text())
        assert not manifest['preview'] and manifest['engine']=='BLENDER_WORKBENCH'
        for name,digest in manifest['source_hashes'].items():
            assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest,name
        frames=manifest['frames'];count=len(frames)
        assert [f['index'] for f in frames]==list(range(count))
        assert count=={'shape':6,'size':49,'aspect':49,'rotation':49,'smoothing':9,'gallery':8}[study]
        if study=='aspect':
            assert [frames[i]['parameters']['outline_aspect_ratio'] for i in (0,24,48)]==[.5,1.,2.4]
        for f in frames:assert all(.04<v<.96 for v in f['bounds']),f
        if count==49:
            order=[0]*24+list(range(49))+[48]*24+list(range(47,0,-1))
        else:
            order=[i for i in range(count) for _ in range(18 if study=='smoothing' else 36)]
        # Keep only one chapter of high-resolution images in memory.
        images=[frame_image(manifest,f) for f in frames]
        assert ImageChops.difference(images[0].crop((0,112,1200,1112)),images[-1].crop((0,112,1200,1112))).getbbox(),study
        path=VIDEOS/f'palm-{study}.mp4'
        encode(path,(images[i] for i in order))
        images[0].save(VIDEOS/f'palm-{study}.png')
        thumb=images[len(images)//2].resize((400,400),Image.Resampling.LANCZOS)
        posters.append(thumb)
        chapters.append(dict(study=study,title=manifest['title'],file=path.name,frames=len(order),seconds=len(order)/FPS))
        print('ENCODED '+json.dumps(chapters[-1]),flush=True)
    concat=VIDEOS/'concat.txt'
    concat.write_text(''.join(f"file '{c['file']}'\n" for c in chapters))
    combined=VIDEOS/'palm-parameters-overview.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(combined)],check=True)
    sheet=Image.new('RGB',(1200,800),'white')
    for i,poster in enumerate(posters):sheet.paste(poster,((i%3)*400,(i//3)*400))
    sheet.save(OUT/'overview.png')
    checks=[]
    for path in [combined]+[VIDEOS/c['file'] for c in chapters]:
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],text=True))
        stream=probe['streams'][0]
        expected=sum(c['frames'] for c in chapters) if path==combined else next(c['frames'] for c in chapters if c['file']==path.name)
        assert (stream['width'],stream['height'])==SIZE
        assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
        assert stream['r_frame_rate']=='24/1' and int(stream['nb_frames'])==expected
        subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(path),'-f','null','-'],check=True)
        checks.append(dict(file=path.name,frames=expected,seconds=float(probe['format']['duration']),bytes=path.stat().st_size,full_decode='passed'))
    (OUT/'video-checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'chapters.json').write_text(json.dumps(chapters,indent=2))
    cards=''.join(f'<article><h2>{html.escape(c["title"])}</h2><video controls preload="none" poster="videos/palm-{c["study"]}.png" src="videos/{c["file"]}"></video><p><a href="videos/{c["file"]}" download>Download MP4</a> · {c["seconds"]:g} seconds</p></article>' for c in chapters)
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Palm parameter video review</title><style>body{max-width:1000px;margin:48px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#18202a;background:white}h1{font-size:28px}h2{font-size:21px}video{display:block;width:100%;max-height:76vh;background:white}a{color:#245889}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:32px}article{margin:16px 0}p{color:#4a515b}</style><h1>Palm parameter studies</h1><p>Original generator geometry and materials. Fixed front view, white background and approved cavity lighting. Palms are shown without finger or thumb mounts to isolate the outline parameters.</p><video controls preload="metadata" poster="videos/palm-shape.png" src="videos/palm-parameters-overview.mp4"></video><p><a href="videos/palm-parameters-overview.mp4" download>Download combined video</a></p><section>'''+cards+'''</section><p>The rotation sequence preserves the generator's normalization to a fixed X width. Smoothing uses a fixed 10 mm outline sampling resolution; other sections use 3 mm.</p></html>''',encoding='utf-8')
    (OUT/'render-notes.txt').write_text('''Palm parameter studies — review export

All geometry is produced by the original generation_v2 Config, Hand and PalmMesh classes.
No source generator files are modified. The caller clears default digit attachment arrays
to generate isolated palms with zero fingers and zero thumbs. These are unmounted palm
outlines, not assembled hands with hidden fingers.

Materials: original generation_v2/blender/components.blend, no palette overrides.
Rendering: Blender Workbench, mylight.sl, Both cavity; World 2.5/2.5, Screen 2/2.
Camera: fixed front orthographic along -Z; white background. Scale 0.28 m,
or 0.36 m throughout the aspect-ratio clip to fit the taller endpoint.
Aspect ratio (width/height): 0.5 -> 1.0 -> 2.4, fixed 130 mm width.
Output: H.264 MP4, 1200 x 1200, 24 fps; combined overview and six separate clips.
Outline rotation uses the native generator's fixed-X-width normalization.
Integer parameters use discrete changes; continuous parameters are regenerated for
every sample, with cosine easing and a forward/reverse loop. No fake mesh morphing.

Reproduce using Blender --background --factory-startup --python-exit-code 1
--python website_assets/scripts/render_palm_overview.py
Then run Python website_assets/scripts/encode_palm_overview.py (Pillow and ffmpeg).

Website and previous hand animations are unchanged. Await review of this video first.
''',encoding='utf-8')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
