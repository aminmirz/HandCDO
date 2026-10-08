"""Render the approved hand with native kinematic configurations simultaneously on every digit."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/simultaneous-kinematics'
BLENDER='C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'

def plan():
    prior=json.loads((ROOT/'preview/thumb-assembly/manifest.json').read_text())
    final=prior['frames'][-1]
    base={k:final[k] for k in ('thumb_location','locations','identities')}
    pools={1:['0-11-','1--12','4-1-2','3-12-1','2--1'],
           2:['3-2-','0--112','1-11-','2-1-12','4--1'],
           3:['4--21','3-12-1','2-2-11','0-1-','1-2-'],
           0:['0--1','1--12','0--211','1--2','0--11']}
    current={1:'2-1-1',2:'2-1-1',3:'2-1-1',0:'1--22'}
    sequences={k:[v] for k,v in current.items()}
    frames=[]
    def add():
        frames.append(dict(base,index=len(frames),stage=1,
                           finger_codes=[current[k] for k in base['identities']],requested_thumb_code=current[0]))
    add()
    for i in range(14):
        for identity,step,shift in ((1,1,1),(2,2,1),(3,3,1),(0,4,2)):
            current[identity]=pools[identity][(i*step+(i//5)*shift)%5]
            sequences[identity].append(current[identity])
        add()
    order=sum(([i]*10 for i in range(15)),[])
    assert len(frames)==15 and len(order)==150
    schedule=dict(frames=frames,order=order,keyframes=[0,4,8,14],fps=30,duration_seconds=5,
                  source_hashes=prior['source_hashes'],baseline=final,sequences=sequences,camera_scale=.66)
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
    requested=schedule['keyframes'] if args.preview else range(15)
    missing=[i for i in requested if not (OUT/'frames'/f'{i:04d}.json').exists()]
    batches=[missing[i:i+8] for i in range(0,len(missing),8)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in as_completed([pool.submit(render,b) for b in batches]):result.result()
    if args.preview:return
    frames=[json.loads((OUT/'frames'/f'{i:04d}.json').read_text()) for i in range(15)]
    baseline=schedule['baseline'];base=np.array(baseline['bases_mm']);normals=np.array(baseline['normals'])
    def joint_types(code,thumb=False):
        rotation,before,after=code.split('-')
        added=lambda text:[{'1':4,'2':7}[digit] for digit in text]
        block=({'0':[3],'1':[3,1]} if thumb else {'0':[],'1':[2,5],'2':[2,6],'3':[3],'4':[2,3]})[rotation]
        return added(before)+block+added(after)
    for i,f in enumerate(frames):
        assert f['thumb_count']==1 and f['finger_count']==3
        assert f['native_finger_codes']==f['finger_codes'] and f['native_thumb_code']==f['requested_thumb_code']
        expected=[joint_types(code) for code in f['finger_codes']]+[joint_types(f['requested_thumb_code'],True)]
        assert f['native_joint_types']==expected,(i,f['native_joint_types'],expected)
        assert np.allclose(f['bases_mm'],base,atol=1e-6) and np.allclose(f['normals'],normals)
        assert np.allclose(f['thumb_base_mm'],baseline['thumb_base_mm']) and np.allclose(f['thumb_normal'],baseline['thumb_normal'])
        assert all(.025<v<.975 for v in f['bounds'])
        if i:
            prev=frames[i-1]
            a=prev['finger_codes']+[prev['requested_thumb_code']];b=f['finger_codes']+[f['requested_thumb_code']]
            assert sum(x!=y for x,y in zip(a,b))==4
    for identity,seq in schedule['sequences'].items():
        assert len({code.split('-')[0] for code in seq})>=2
        assert len({code.split('-')[2] for code in seq})>=2
        if identity!=0:assert len({code.split('-')[1] for code in seq})>=2
        counts=[len(joint_types(code,identity==0)) for code in seq]
        delta=np.diff(counts)
        assert np.count_nonzero(delta>0)>=2 and np.count_nonzero(delta<0)>=2
        added=[len(code.split('-')[1])+len(code.split('-')[2]) for code in seq]
        assert np.any(np.diff(added)>0) and np.any(np.diff(added)<0)
    for name,digest in schedule['source_hashes'].items():assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest
    fonts=[ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n) for n in (32,26,20)]
    @lru_cache(maxsize=4)
    def image(index):
        raw=Image.open(OUT/'frames'/f'{index:04d}.png').convert('RGB');assert raw.size==(1200,1000)
        assert all(min(raw.getpixel(p))>=250 for p in ((0,0),(1199,0),(0,999),(1199,999)))
        canvas=Image.new('RGB',(1200,1200),'white');canvas.paste(raw,(0,112));draw=ImageDraw.Draw(canvas)
        f=frames[index]
        draw.text((48,24),'Kinematic configurations',font=fonts[0],fill='#18202a')
        codes=dict(zip(f['identities'],f['finger_codes']))
        caption='Rotation types and added joints change across all digits'
        draw.text((48,68),caption,font=fonts[1],fill='#4a515b')
        draw.line((48,108,1152,108),fill='#e7e9eb')
        draw.text((48,1125),f"Left: {codes[1]} | Middle: {codes[2]} | Right: {codes[3]} | Thumb: {f['requested_thumb_code']}",font=fonts[2],fill='#626a74')
        draw.text((48,1158),'Code: rotation-before-after | Added joints: 1 = short, 2 = long',font=fonts[2],fill='#626a74')
        bar=1200/schedule['camera_scale']*.02;draw.line((1040,1142,1040+bar,1142),fill='#626a74',width=2)
        for x in (1040,1040+bar):draw.line((x,1137,x,1147),fill='#626a74',width=2)
        draw.text((1038,1154),'20 mm',font=fonts[2],fill='#626a74')
        return canvas
    video=OUT/'simultaneous-kinematics-5s.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1200x1200','-r','30','-i','-',
             '-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    try:
        for index in schedule['order']:process.stdin.write(image(index).tobytes())
        process.stdin.close();assert process.wait()==0
    finally:
        if process.poll() is None:process.kill()
    image(8).save(OUT/'poster.png')
    sheet=Image.new('RGB',(1200,1200),'white')
    for n,i in enumerate((0,4,8,14)):
        sheet.paste(image(i).resize((600,600),Image.Resampling.LANCZOS),((n%2)*600,(n//2)*600))
    sheet.save(OUT/'stages.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
    stream=probe['streams'][0]
    assert int(stream['nb_frames'])==150 and stream['r_frame_rate']=='30/1' and abs(float(probe['format']['duration'])-5)<.01
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p' and (stream['width'],stream['height'])==(1200,1200)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    for name,digest in preserved.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    checks=dict(seconds=5,frames=150,native_configurations=15,distinct_finger_sequences='passed',
                native_codes_and_joint_types='passed',rotation_and_before_after_variation='passed',all_digits_change_together='passed',repeated_joint_additions_and_removals='passed',
                fixed_bases_and_angles='passed',framing='passed',full_decode='passed',previous_videos_unchanged=True,
                source_hashes='passed',bytes=video.stat().st_size)
    (OUT/'video-checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'manifest.json').write_text(json.dumps({**schedule,'frames':frames,'preserved_videos':preserved,
        'engine':'BLENDER_WORKBENCH','materials':'Original components.blend','light':'mylight.sl',
        'cavity':{'type':'BOTH','world':[2.5,2.5],'screen':[2,2]}},indent=2))
    shutil.copyfile(ROOT/'preview/thumb-orbit/mylight.sl',OUT/'mylight.sl')
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Kinematic configurations</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#18202a;background:white}h1{font-size:28px}video{display:block;width:100%;max-height:78vh}a{color:#245889}</style><h1>Kinematic configurations</h1><p>Each finger has a different sequence of rotation types and short or long joints before and after the rotation block. The thumb also changes between its native rotation modes and after-joint configurations. All digits change together every third of a second, repeatedly adding and removing joints; base positions stay fixed.</p><video controls loop preload="metadata" poster="poster.png" src="simultaneous-kinematics-5s.mp4"></video><p><a href="simultaneous-kinematics-5s.mp4" download>Download the 5-second video</a></p></html>''',encoding='utf-8')
    (OUT/'render-notes.txt').write_text('Original generation_v2 native FingerCode2Chain and ThumbCode2Chain. All three fingers and the thumb change at every transition, with distinct sequences per digit. Rotation types and added short/long joints before and after vary; thumb uses native R--AFTER grammar. Every digit repeatedly gains and loses joints. Fifteen configurations held 10 frames each: 150 frames, 5 seconds at 30 fps. Fixed palm and mounts; fixed front camera scale 0.66. Original materials and mylight.sl, white background, Both cavity world 2.5/2.5 and screen 2/2. Previous videos preserved. Reproduce using build_simultaneous_kinematics.py.\n')
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
