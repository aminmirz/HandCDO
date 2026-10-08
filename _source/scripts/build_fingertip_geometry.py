"""Render the approved hand with independent fingertip dimensions on every finger and thumb."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/fingertip-geometry'
BLENDER='C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'

def plan():
    prior=json.loads((ROOT/'preview/thumb-assembly/manifest.json').read_text())
    final=prior['frames'][-1]
    base={k:final[k] for k in ('thumb_location','locations','identities')}
    def scale(t,identity):
        envelope=math.sin(math.pi*t)**2
        return [1+envelope*(.3+.6*math.sin(2*math.pi*t+identity*1.27+axis*1.73)) for axis in range(3)]
    frames=[]
    for i in range(101):
        t=i/100
        frames.append(dict(base,index=i,stage=1,progress=t,finger_tip_scales=[scale(t,k) for k in base['identities']],
                           thumb_tip_scale=scale(t,0)))
    order=[round(i*100/149) for i in range(150)]
    schedule=dict(frames=frames,order=order,keyframes=[0,25,50,75,100],fps=30,duration_seconds=5,
                  source_hashes=prior['source_hashes'],baseline=final,camera_scale=.54)
    (OUT/'plan.json').write_text(json.dumps(schedule,indent=2))
    return schedule

def render(indices):
    if not indices:return
    temp=OUT/'tmp'/f'process-{indices[0]}';temp.mkdir(parents=True,exist_ok=True)
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    path=logs/f'batch-{indices[0]:04d}.log'
    with path.open('w',encoding='utf-8') as log:
        result=subprocess.run([BLENDER,'-b','--factory-startup','-t','4','--python-exit-code','1','--python',
            str(ROOT/'scripts/render_thumb_assembly.py'),'--','--output',str(OUT),'--indices',*map(str,indices)],
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TEMP=str(temp),TMP=str(temp)),
            stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:raise RuntimeError(path.read_text(encoding='utf-8',errors='replace')[-5000:])
    print(f'BATCH_COMPLETE {indices[0]}-{indices[-1]}',flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    preserved={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (ROOT/'preview').rglob('*.mp4') if OUT not in p.parents}
    schedule=plan()
    requested=schedule['keyframes'] if args.preview else range(101)
    missing=[i for i in requested if not (OUT/'frames'/f'{i:04d}.json').exists()]
    batches=[missing[i:i+15] for i in range(0,len(missing),15)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in as_completed([pool.submit(render,b) for b in batches]):result.result()
    if args.preview:return
    frames=[json.loads((OUT/'frames'/f'{i:04d}.json').read_text()) for i in range(101)]
    baseline=schedule['baseline'];base=np.array(baseline['bases_mm']);normals=np.array(baseline['normals'])
    for f in frames:
        assert f['thumb_count']==1 and f['finger_count']==3 and f['thumb_code']=='1--22' and f['finger_code']=='2-1-1'
        assert np.allclose(f['bases_mm'],base,atol=1e-6) and np.allclose(f['normals'],normals)
        assert np.allclose(f['thumb_base_mm'],baseline['thumb_base_mm']) and np.allclose(f['thumb_normal'],baseline['thumb_normal'])
        requested=np.array(f['finger_tip_scales']+[f['thumb_tip_scale']])
        assert np.allclose(f['native_tip_scales'],requested)
        assert np.allclose(f['tip_attachment_matrices'],frames[0]['tip_attachment_matrices'])
        measured=np.array(f['tip_dimensions_local_mm'])/np.array(frames[0]['tip_dimensions_local_mm'])
        assert np.allclose(measured,requested,atol=.015),(f['index'],measured,requested)
        assert all(.025<v<.975 for v in f['bounds'])
    tracks=np.array([sum(f['finger_tip_scales'],[])+f['thumb_tip_scale'] for f in frames])
    assert tracks.shape==(101,12) and np.allclose(tracks[0],1) and np.allclose(tracks[-1],1)
    assert np.all(np.ptp(tracks,axis=0)>.25)
    assert len(set(np.argmax(tracks,axis=0)))>=8
    for name,digest in schedule['source_hashes'].items():assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest
    fonts=[ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n) for n in (32,26,20)]
    @lru_cache(maxsize=4)
    def image(index):
        raw=Image.open(OUT/'frames'/f'{index:04d}.png').convert('RGB');assert raw.size==(1200,1000)
        assert all(min(raw.getpixel(p))>=250 for p in ((0,0),(1199,0),(0,999),(1199,999)))
        canvas=Image.new('RGB',(1200,1200),'white');canvas.paste(raw,(0,112));draw=ImageDraw.Draw(canvas)
        f=frames[index]
        draw.text((48,24),'Fingertip geometry',font=fonts[0],fill='#18202a')
        draw.text((48,68),'Independent fingertip length, width and thickness',font=fonts[1],fill='#4a515b')
        draw.line((48,108,1152,108),fill='#e7e9eb')
        draw.text((48,1125),'Original fingertip meshes | Independent scaling along all three local axes',font=fonts[2],fill='#626a74')
        draw.text((48,1158),'Palm, joints and mounting positions fixed',font=fonts[2],fill='#626a74')
        bar=1200/.54*.02;draw.line((1040,1142,1040+bar,1142),fill='#626a74',width=2)
        for x in (1040,1040+bar):draw.line((x,1137,x,1147),fill='#626a74',width=2)
        draw.text((1038,1154),'20 mm',font=fonts[2],fill='#626a74')
        return canvas
    video=OUT/'fingertip-geometry-5s.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1200x1200','-r','30','-i','-',
             '-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    try:
        for index in schedule['order']:process.stdin.write(image(index).tobytes())
        process.stdin.close();assert process.wait()==0
    finally:
        if process.poll() is None:process.kill()
    image(50).save(OUT/'poster.png')
    sheet=Image.new('RGB',(1200,1200),'white')
    for n,i in enumerate((0,25,50,75)):
        sheet.paste(image(i).resize((600,600),Image.Resampling.LANCZOS),((n%2)*600,(n//2)*600))
    sheet.save(OUT/'stages.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
    stream=probe['streams'][0]
    assert int(stream['nb_frames'])==150 and stream['r_frame_rate']=='30/1' and abs(float(probe['format']['duration'])-5)<.01
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p' and (stream['width'],stream['height'])==(1200,1200)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    for name,digest in preserved.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    checks=dict(seconds=5,frames=150,native_renders=101,independent_scale_channels=12,
                native_tip_scales='passed',actual_mesh_dimensions='passed',fixed_tip_attachments='passed',
                fixed_bases_and_angles='passed',framing='passed',full_decode='passed',
                previous_videos_unchanged=True,source_hashes='passed',bytes=video.stat().st_size)
    (OUT/'video-checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'manifest.json').write_text(json.dumps({**schedule,'frames':frames,'preserved_videos':preserved,
        'engine':'BLENDER_WORKBENCH','materials':'Original components.blend','light':'mylight.sl',
        'cavity':{'type':'BOTH','world':[2.5,2.5],'screen':[2,2]}},indent=2))
    shutil.copyfile(ROOT/'preview/thumb-orbit/mylight.sl',OUT/'mylight.sl')
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Fingertip geometry</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#18202a;background:white}h1{font-size:28px}video{display:block;width:100%;max-height:78vh}a{color:#245889}</style><h1>Fingertip geometry</h1><p>Fingertip length, width and thickness vary independently on each finger and the thumb using the original generator scaling parameters. Palm, joints and mounting positions stay fixed.</p><video controls loop preload="metadata" poster="poster.png" src="fingertip-geometry-5s.mp4"></video><p><a href="fingertip-geometry-5s.mp4" download>Download the 5-second video</a></p></html>''',encoding='utf-8')
    (OUT/'render-notes.txt').write_text('Original generation_v2 FingerConfig.fingertip_scale_factor on the three 2-1-1 fingers and 1--22 thumb. All twelve local-axis scale channels follow different phases and return to unit scale. Native components.blend fingertip meshes, native scaling and insert-hole booleans. Actual local mesh extents checked against requested scale ratios, with attachment transforms fixed. Front orthographic camera at (0,0,1), scale 0.54, original materials, white background and approved mylight.sl/Both cavity. 101 native renders sampled into 150 frames, 5 seconds at 30 fps. Previous videos preserved. Reproduce using build_fingertip_geometry.py.\n')
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
