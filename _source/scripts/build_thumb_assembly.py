"""Render and encode a separate thumb orbit followed by three finger additions."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/thumb-assembly'
BLENDER='C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'

def render(indices):
    temp=OUT/'tmp'/f'process-{indices[0]}';temp.mkdir(parents=True,exist_ok=True)
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    path=logs/f'batch-{indices[0]:04d}.log'
    with path.open('w',encoding='utf-8') as log:
        result=subprocess.run([BLENDER,'-b','--factory-startup','-t','4','--python-exit-code','1','--python',
            str(ROOT/'scripts/render_thumb_assembly.py'),'--','--indices',*map(str,indices)],
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TEMP=str(temp),TMP=str(temp)),
            stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:raise RuntimeError(path.read_text(encoding='utf-8',errors='replace')[-5000:])
    print(f'BATCH_COMPLETE {indices[0]}-{indices[-1]}',flush=True)

def main():
    preserved={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (ROOT/'preview').rglob('*.mp4') if OUT not in p.parents}
    schedule=json.loads((OUT/'plan.json').read_text())
    (OUT/'frames').mkdir(exist_ok=True)
    # Reuse the exact approved orbit renders, sampled at twice the speed.
    for f in schedule['frames'][:91]:
        src=ROOT/'preview/thumb-orbit/frames'/f'{f["orbit_source"]:04d}'
        dst=OUT/'frames'/f'{f["index"]:04d}'
        shutil.copyfile(src.with_suffix('.png'),dst.with_suffix('.png'))
        original=json.loads(src.with_suffix('.json').read_text())
        dst.with_suffix('.json').write_text(json.dumps({**original,**f,'thumb_base_mm':original['base_mm'],
            'thumb_normal':original['normal'],'thumb_code':original['code']},indent=2))
    missing=[f['index'] for f in schedule['frames'][91:] if not (OUT/'frames'/f'{f["index"]:04d}.json').exists()]
    batches=[missing[i:i+18] for i in range(0,len(missing),18)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in as_completed([pool.submit(render,b) for b in batches]):result.result()
    frames=[json.loads((OUT/'frames'/f'{i:04d}.json').read_text()) for i in range(184)]
    for i,f in enumerate(frames):
        assert f['thumb_count']==1 and f['thumb_code']=='1--22'
        assert f['finger_count']==(0 if i<91 else 1+(i-91)//31)
        assert all(.025<v<.975 for v in f['bounds'])
        if i>=91:
            assert np.allclose(f['thumb_base_mm'],frames[90]['thumb_base_mm'])
            assert np.allclose(f['thumb_normal'],[1,0])
            if len(f['bases_mm'])>1:
                assert min(np.linalg.norm(np.array(a)-b) for j,a in enumerate(f['bases_mm']) for b in f['bases_mm'][j+1:])>=40.5
    assert np.allclose(frames[-1]['bases_mm'],[[45,130/1.55/2],[0,130/1.55/2],[-45,130/1.55/2]])
    assert np.allclose(frames[-1]['normals'],[[0,1]]*3)
    orbit=np.unwrap(np.array([f['thumb_location'] for f in frames[:91]])*2*np.pi)
    assert abs((orbit[-1]-orbit[0])/(2*np.pi)-1)<1e-6
    for name,digest in schedule['source_hashes'].items():assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest
    fonts=[ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n) for n in (32,26,20)]
    @lru_cache(maxsize=4)
    def image(index):
        raw=Image.open(OUT/'frames'/f'{index:04d}.png').convert('RGB');assert raw.size==(1200,1000)
        assert all(min(raw.getpixel(p))>=250 for p in ((0,0),(1199,0),(0,999),(1199,999)))
        canvas=Image.new('RGB',(1200,1200),'white');canvas.paste(raw,(0,112));draw=ImageDraw.Draw(canvas)
        f=frames[index];count=f['finger_count']
        draw.text((48,24),'Thumb and finger placement',font=fonts[0],fill='#18202a')
        caption='Thumb: full palm boundary' if count==0 else f'Adding finger {count} of 3'
        draw.text((48,68),caption,font=fonts[1],fill='#4a515b');draw.line((48,108,1152,108),fill='#e7e9eb')
        draw.text((48,1155),'Thumb: 1--22 | Fingers: 2-1-1 | Palm size: 130 mm',font=fonts[2],fill='#626a74')
        bar=1200/.54*.02;draw.line((1040,1142,1040+bar,1142),fill='#626a74',width=2)
        for x in (1040,1040+bar):draw.line((x,1137,x,1147),fill='#626a74',width=2)
        draw.text((1038,1154),'20 mm',font=fonts[2],fill='#626a74')
        return canvas
    video=OUT/'thumb-and-fingers-7s.mp4'
    command=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1200x1200','-r','30','-i','-',
             '-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for index in schedule['order']:process.stdin.write(image(index).tobytes())
        process.stdin.close();assert process.wait()==0
    finally:
        if process.poll() is None:process.kill()
    image(183).save(OUT/'poster.png')
    sheet=Image.new('RGB',(1200,1200),'white')
    for n,i in enumerate((90,121,152,183)):sheet.paste(image(i).resize((600,600),Image.Resampling.LANCZOS),((n%2)*600,(n//2)*600))
    sheet.save(OUT/'stages.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
    stream=probe['streams'][0]
    assert int(stream['nb_frames'])==210 and stream['r_frame_rate']=='30/1' and abs(float(probe['format']['duration'])-7)<.01
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    for name,digest in preserved.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    checks=dict(seconds=7,frames=210,full_thumb_circuit=True,final_thumb_count=1,final_finger_count=3,
                fixed_thumb_after_orbit=True,final_top_finger_normals='passed',framing='passed',full_decode='passed',
                previous_videos_unchanged=True,source_hashes='passed',bytes=video.stat().st_size)
    (OUT/'video-checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'manifest.json').write_text(json.dumps({**schedule,'frames':frames,'preserved_videos':preserved,
        'engine':'BLENDER_WORKBENCH','materials':'Original components.blend','light':'mylight.sl',
        'cavity':{'type':'BOTH','world':[2.5,2.5],'screen':[2,2]}},indent=2))
    shutil.copyfile(ROOT/'preview/thumb-orbit/mylight.sl',OUT/'mylight.sl')
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Thumb and finger placement</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#18202a;background:white}h1{font-size:28px}video{display:block;width:100%;max-height:78vh}a{color:#245889}</style><h1>Thumb and finger placement</h1><p>The thumb travels around the full palm boundary, settles on the right, and three fingers are added along the top.</p><video controls loop preload="metadata" poster="poster.png" src="thumb-and-fingers-7s.mp4"></video><p><a href="thumb-and-fingers-7s.mp4" download>Download the 7-second video</a></p></html>''',encoding='utf-8')
    (OUT/'render-notes.txt').write_text('Original generation_v2 thumb 1--22 and fingers 2-1-1. Approved thumb orbit frames reused at twice their previous playback speed, followed by three sequential finger additions. Total 7 seconds. Native palm, mount, digit geometry and materials; full-boundary attachment points supplied by this local rendering caller. White background, original mylight.sl, Both cavity world 2.5/2.5 and screen 2/2. Front orthographic camera scale 0.54. Previous videos unchanged. Reproduce: render_thumb_assembly.py --plan in Blender, then build_thumb_assembly.py in Python.\n')
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
