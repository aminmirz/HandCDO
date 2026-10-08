"""Render, encode and validate a separate actual-thumb orbit video."""
import sys
sys.dont_write_bytecode=True
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/thumb-orbit'
BLENDER='C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'

def render(indices):
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    temp=OUT/'tmp'/f'process-{indices[0]:04d}';temp.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TEMP=str(temp),TMP=str(temp))
    command=[BLENDER,'-b','--factory-startup','-t','4','--python-exit-code','1',
             '--python',str(ROOT/'scripts/render_thumb_orbit.py'),'--','--indices',*map(str,indices)]
    log=logs/f'batch-{indices[0]:04d}.log'
    with log.open('w',encoding='utf-8') as handle:
        result=subprocess.run(command,env=env,stdout=handle,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:raise RuntimeError(log.read_text(encoding='utf-8',errors='replace')[-4000:])
    print(f'BATCH_COMPLETE {indices[0]}-{indices[-1]}',flush=True)

def main():
    plan=json.loads((OUT/'plan.json').read_text())
    old_videos=[ROOT/'preview'/p for p in ('finger-assembly/finger-assembly-7s.mp4',
                'finger-offset/finger-offset-7s.mp4','finger-location/finger-location-7s.mp4',
                'palm-parameters/videos/palm-parameters-overview-7s.mp4')]
    preserved={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in old_videos}
    missing=[]
    for f in plan['frames']:
        path=OUT/'frames'/f'{f["index"]:04d}.png';meta=path.with_suffix('.json')
        if not(path.exists() and meta.exists() and json.loads(meta.read_text()).get('code')==plan['code']):missing.append(f['index'])
    batches=[missing[i:i+18] for i in range(0,len(missing),18)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(render,b) for b in batches]):future.result()
    frames=[json.loads((OUT/'frames'/f'{f["index"]:04d}.json').read_text()) for f in plan['frames']]
    manifest={**plan,'frames':frames,'engine':'BLENDER_WORKBENCH','materials':'Original components.blend thumb and palm materials',
              'light':'mylight.sl','cavity':{'type':'BOTH','world':[2.5,2.5],'screen':[2,2]},
              'camera':{'position':[0,0,1],'front_axis':'-Z','scale_m':.54},'preserved_videos':preserved}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copyfile(ROOT/'preview/front-cavity-horizontal-thumb/mylight.sl',OUT/'mylight.sl')
    assert len(frames)==181 and len(plan['order'])==210
    orbit=np.unwrap(np.array([f['location'] for f in frames])*2*np.pi)
    assert abs((orbit[-1]-orbit[0])/(2*np.pi)-1)<1e-6
    assert np.allclose(frames[0]['point_mm'],frames[-1]['point_mm'],atol=1e-6)
    assert np.allclose(frames[0]['base_mm'],frames[-1]['base_mm'],atol=1e-6)
    for f in frames:
        assert f['code']==plan['code'] and f['thumb_count']==1 and f['finger_count']==0
        assert all(.025<v<.975 for v in f['bounds'])
    for name,digest in plan['source_hashes'].items():
        assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest
    fonts=[ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n) for n in (32,26,20)]
    @lru_cache(maxsize=4)
    def image(index):
        f=frames[index]
        raw=Image.open(OUT/'frames'/f'{index:04d}.png').convert('RGB')
        assert raw.size==(1200,1000)
        assert all(min(raw.getpixel(p))>=250 for p in ((0,0),(1199,0),(0,999),(1199,999)))
        result=Image.new('RGB',(1200,1200),'white');result.paste(raw,(0,112))
        draw=ImageDraw.Draw(result)
        draw.text((48,24),'Thumb placement',font=fonts[0],fill='#18202a')
        draw.text((48,68),f'Full palm boundary | Location: {f["location"]:.3f}',font=fonts[1],fill='#4a515b')
        draw.line((48,108,1152,108),fill='#e7e9eb',width=1)
        draw.text((48,1155),f'Thumb configuration: {plan["code"]} | Palm size: 130 mm',font=fonts[2],fill='#626a74')
        bar=1200/.54*.02
        draw.line((1040,1142,1040+bar,1142),fill='#626a74',width=2)
        for x in (1040,1040+bar):draw.line((x,1137,x,1147),fill='#626a74',width=2)
        draw.text((1038,1154),'20 mm',font=fonts[2],fill='#626a74')
        return result
    video=OUT/'thumb-orbit-7s.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1200x1200',
        '-r','30','-i','-','-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    try:
        for index in plan['order']:process.stdin.write(image(index).tobytes())
        process.stdin.close()
        if process.wait()!=0:raise RuntimeError('Encoding failed.')
    finally:
        if process.poll() is None:process.kill()
    image(0).save(OUT/'poster.png')
    sheet=Image.new('RGB',(1200,1200),'white')
    for n,index in enumerate((0,60,90,120)):
        sheet.paste(image(index).resize((600,600),Image.Resampling.LANCZOS),((n%2)*600,(n//2)*600))
    sheet.save(OUT/'views.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
    stream=probe['streams'][0]
    assert int(stream['nb_frames'])==210 and stream['r_frame_rate']=='30/1'
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p'
    assert (stream['width'],stream['height'])==(1200,1200)
    assert abs(float(probe['format']['duration'])-7)<.01
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    for name,digest in preserved.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    checks=dict(file=video.name,seconds=7,fps=30,frames=210,native_renders=181,thumb_configuration=plan['code'],
                thumb_count=1,finger_count=0,full_boundary_circuits=1,closed_loop='passed',framing='passed',
                source_hashes='passed',previous_videos_unchanged=True,full_decode='passed',bytes=video.stat().st_size)
    (OUT/'video-checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Thumb placement</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#18202a;background:white}h1{font-size:28px}video{display:block;width:100%;max-height:78vh}a{color:#245889}</style><h1>Thumb placement</h1><p>The original thumb configuration travels around the full palm boundary and returns to its starting position.</p><video controls loop preload="metadata" poster="poster.png" src="thumb-orbit-7s.mp4"></video><p><a href="thumb-orbit-7s.mp4" download>Download the separate 7-second video</a></p><p>Original thumb and palm geometry, materials, front view and cavity lighting. The rendering script supplies full-boundary positions; the original generator builds the thumb mount and chooses its left/right configuration.</p></html>''',encoding='utf-8')
    (OUT/'render-notes.txt').write_text(f'''Separate thumb orbit

Actual thumb chain {plan['code']}, taken from the approved horizontal-thumb model.
Zero fingers, one thumb, 130 mm palm, aspect ratio 1.55. Original thumb/palm
materials, white background and approved Workbench cavity settings.

The add-on UI restricts thumb placement to its lower valid region. This local
rendering caller supplies one point on the full closed original palm outline,
then calls the unchanged native Hand, ThumbCode2Chain, FingerAssembly and
PalmMesh implementations. The original generator selects left/right thumb
configuration according to the attachment point's side. Its original thumb
base-angle parameter aligns the base with the outward boundary normal.
This is a generated embodiment sequence, not a physically moving attachment.

181 native renders, 210 output frames at 30 fps, duration 7 seconds.
Reproduce: run render_thumb_orbit.py --plan inside Blender, then normal Python
build_thumb_orbit.py. All output remains in website_assets/preview/thumb-orbit.
Previous videos are kept intact and their hashes are checked after rendering.
''',encoding='utf-8')
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
