"""Render the approved hand with each finger's native normal offset swept separately."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/individual-finger-offsets'
BLENDER='C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'

def plan():
    prior=json.loads((ROOT/'preview/thumb-assembly/manifest.json').read_text())
    final=prior['frames'][-1]
    base={k:final[k] for k in ('thumb_location','locations','identities')}
    frames=[dict(base,index=0,stage=0,active_identity=0,normal_offsets_mm=[0.,0.,0.])]
    order=[dict(index=0,active_identity=1)]*9
    for identity in (1,2,3):
        indices=[]
        for step in range(1,31):
            offsets=[0.,0.,0.];offsets[base['identities'].index(identity)]=25*(1-math.cos(math.pi*step/30))/2
            index=len(frames);indices.append(index)
            frames.append(dict(base,index=index,stage=identity,active_identity=identity,normal_offsets_mm=offsets))
        cycle=indices+list(reversed(indices[:-1]))+[0]+[0]*7
        order.extend(dict(index=i,active_identity=identity) for i in cycle)
    assert len(frames)==91 and len(order)==210
    schedule=dict(frames=frames,order=order,keyframes=[0,30,60,90],fps=30,duration_seconds=7,
                  source_hashes=prior['source_hashes'],baseline=final,max_offset_mm=25.)
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
    requested=schedule['keyframes'] if args.preview else range(91)
    missing=[i for i in requested if not (OUT/'frames'/f'{i:04d}.json').exists()]
    batches=[missing[i:i+15] for i in range(0,len(missing),15)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in as_completed([pool.submit(render,b) for b in batches]):result.result()
    if args.preview:return
    frames=[json.loads((OUT/'frames'/f'{i:04d}.json').read_text()) for i in range(91)]
    baseline=schedule['baseline'];base=np.array(baseline['bases_mm']);normals=np.array(baseline['normals'])
    for f in frames:
        offsets=np.array(f['normal_offsets_mm'])
        assert np.count_nonzero(offsets)>0 if f['index'] else np.count_nonzero(offsets)==0
        assert np.count_nonzero(offsets)<=1 and offsets.min()>=0 and offsets.max()<=25
        assert f['thumb_count']==1 and f['finger_count']==3 and f['thumb_code']=='1--22' and f['finger_code']=='2-1-1'
        assert np.allclose(f['bases_mm'],base+offsets[:,None]*normals,atol=1e-6)
        assert np.allclose(f['normals'],normals) and np.allclose(f['thumb_base_mm'],baseline['thumb_base_mm'])
        assert np.allclose(f['thumb_normal'],baseline['thumb_normal'])
        assert all(.025<v<.975 for v in f['bounds'])
    for identity,index in enumerate((30,60,90),start=1):
        assert frames[index]['normal_offsets_mm'][baseline['identities'].index(identity)]==25
    for name,digest in schedule['source_hashes'].items():assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest
    fonts=[ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n) for n in (32,26,20)]
    labels={1:'Left finger',2:'Middle finger',3:'Right finger'}
    @lru_cache(maxsize=4)
    def image(index,identity):
        raw=Image.open(OUT/'frames'/f'{index:04d}.png').convert('RGB');assert raw.size==(1200,1000)
        assert all(min(raw.getpixel(p))>=250 for p in ((0,0),(1199,0),(0,999),(1199,999)))
        canvas=Image.new('RGB',(1200,1200),'white');canvas.paste(raw,(0,112));draw=ImageDraw.Draw(canvas)
        offset=frames[index]['normal_offsets_mm'][baseline['identities'].index(identity)]
        draw.text((48,24),'Individual finger base offsets',font=fonts[0],fill='#18202a')
        draw.text((48,68),f'{labels[identity]} | Normal offset: {offset:.1f} mm',font=fonts[1],fill='#4a515b')
        draw.line((48,108,1152,108),fill='#e7e9eb')
        draw.text((48,1155),'Thumb: 1--22 | Fingers: 2-1-1 | Palm size: 130 mm',font=fonts[2],fill='#626a74')
        bar=1200/.54*.02;draw.line((1040,1142,1040+bar,1142),fill='#626a74',width=2)
        for x in (1040,1040+bar):draw.line((x,1137,x,1147),fill='#626a74',width=2)
        draw.text((1038,1154),'20 mm',font=fonts[2],fill='#626a74')
        return canvas
    video=OUT/'individual-finger-offsets-7s.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1200x1200','-r','30','-i','-',
             '-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    try:
        for f in schedule['order']:process.stdin.write(image(f['index'],f['active_identity']).tobytes())
        process.stdin.close();assert process.wait()==0
    finally:
        if process.poll() is None:process.kill()
    image(30,1).save(OUT/'poster.png')
    sheet=Image.new('RGB',(1200,1200),'white')
    for n,(i,identity) in enumerate(((0,1),(30,1),(60,2),(90,3))):
        sheet.paste(image(i,identity).resize((600,600),Image.Resampling.LANCZOS),((n%2)*600,(n//2)*600))
    sheet.save(OUT/'stages.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
    stream=probe['streams'][0]
    assert int(stream['nb_frames'])==210 and stream['r_frame_rate']=='30/1' and abs(float(probe['format']['duration'])-7)<.01
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p' and (stream['width'],stream['height'])==(1200,1200)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    for name,digest in preserved.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    checks=dict(seconds=7,frames=210,native_renders=91,sequence=['left','middle','right'],offset_range_mm=[0,25],
                one_finger_at_a_time='passed',native_normal_offsets='passed',fixed_thumb='passed',
                framing='passed',full_decode='passed',previous_videos_unchanged=True,source_hashes='passed',bytes=video.stat().st_size)
    (OUT/'video-checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'manifest.json').write_text(json.dumps({**schedule,'frames':frames,'preserved_videos':preserved,
        'engine':'BLENDER_WORKBENCH','materials':'Original components.blend','light':'mylight.sl',
        'cavity':{'type':'BOTH','world':[2.5,2.5],'screen':[2,2]}},indent=2))
    shutil.copyfile(ROOT/'preview/thumb-orbit/mylight.sl',OUT/'mylight.sl')
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Individual finger base offsets</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#18202a;background:white}h1{font-size:28px}video{display:block;width:100%;max-height:78vh}a{color:#245889}</style><h1>Individual finger base offsets</h1><p>The left, middle and right fingers each move from 0 to 25 mm normal offset and return, one at a time. The thumb remains fixed.</p><video controls loop preload="metadata" poster="poster.png" src="individual-finger-offsets-7s.mp4"></video><p><a href="individual-finger-offsets-7s.mp4" download>Download the 7-second video</a></p></html>''',encoding='utf-8')
    (OUT/'render-notes.txt').write_text('Original generation_v2 thumb 1--22 and three fingers 2-1-1, from the approved thumb-assembly final frame. Each native finger_base_normal_offset_mm_list parameter separately follows 0 to 25 to 0 mm, left then middle then right. Native palm mount surfaces regenerate with each offset; nominal palm size, outline settings, attachment locations and thumb stay fixed. Original materials, white background, front orthographic camera, mylight.sl, Both cavity world 2.5/2.5 and screen 2/2. 7 seconds at 30 fps; 91 unique native frames. Previous videos preserved. Reproduce using build_individual_offsets.py.\n')
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
