"""Render the approved hand with native contact surface parameters independently."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/contact-surfaces-stronger'
BLENDER='C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'

def plan():
    prior=json.loads((ROOT/'preview/thumb-assembly/manifest.json').read_text())
    final=prior['frames'][-1]
    base={k:final[k] for k in ('thumb_location','locations','identities')}
    def surface(t,identity,pad=0,palm=False):
        phase=identity*.87+pad*1.19
        wave=lambda offset:(1+math.sin(2*math.pi*t+phase+offset))/2
        envelope=math.sin(math.pi*t)**2
        height=(30 if palm else 10)*envelope*(.45+.55*wave(.6))
        return dict(height_mm=height,height_intensity=[1.,.65],
                    spread=[.15+.20*wave(1.),.16+.13*wave(2.4)],
                    aspect_ratio=[.55+1.45*wave(2.),.65+1.2*wave(.5)],
                    rotation_deg=[180*wave(.2),180*wave(2.7)],
                    center_angle_deg=[360*t+phase*57.3,180+360*t-phase*35],
                    center_offset=[.12+.22*wave(3.),.20+.18*wave(.4)])
    frames=[]
    for i in range(101):
        t=i/100
        frames.append(dict(base,index=i,stage=1,progress=t,render_revision='upright_smooth_intensity_2p5',palm_surface=surface(t,5,palm=True),
                           finger_surfaces=[[surface(t,k,p) for p in range(5)] for k in base['identities']],
                           thumb_surfaces=[surface(t,0,p) for p in range(4)]))
    order=[round(i*100/149) for i in range(150)]
    schedule=dict(frames=frames,order=order,keyframes=[0,50,100],fps=30,duration_seconds=5,
                  source_hashes=prior['source_hashes'],baseline=final,camera_scale=.5,
                  camera_target=[.045,.050,0],camera_offset=[.35,-.25,1.])
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
    missing=[i for i in requested if not (OUT/'frames'/f'{i:04d}.json').exists() or
             json.loads((OUT/'frames'/f'{i:04d}.json').read_text()).get('render_revision')!='upright_smooth_intensity_2p5']
    batches=[missing[i:i+10] for i in range(0,len(missing),10)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in as_completed([pool.submit(render,b) for b in batches]):result.result()
    if args.preview:return
    frames=[json.loads((OUT/'frames'/f'{i:04d}.json').read_text()) for i in range(101)]
    baseline=schedule['baseline'];base=np.array(baseline['bases_mm']);normals=np.array(baseline['normals'])
    for f in frames:
        assert f['thumb_count']==1 and f['finger_count']==3 and f['thumb_code']=='1--22' and f['finger_code']=='2-1-1'
        assert np.allclose(f['bases_mm'],base,atol=1e-6) and np.allclose(f['normals'],normals)
        assert np.allclose(f['thumb_base_mm'],baseline['thumb_base_mm']) and np.allclose(f['thumb_normal'],baseline['thumb_normal'])
        assert all(.025<v<.975 for v in f['bounds'])
        assert f['surface_meshes'] and all(v['vertices']>100 for v in f['surface_meshes'].values())
    names=set(frames[0]['surface_meshes'])
    assert all(set(f['surface_meshes'])==names for f in frames)
    changes={name:max(f['surface_meshes'][name]['local_z_extent_mm'] for f in frames)-
                  min(f['surface_meshes'][name]['local_z_extent_mm'] for f in frames) for name in names}
    assert all(delta>.1 for delta in changes.values()),changes
    assert any('f1_' in n for n in names) and any('f2_' in n for n in names) and any('f3_' in n for n in names)
    assert any('t1_' in n for n in names) and 'viz_pad_PalmBody' in names
    for name in names:
        assert abs(frames[0]['surface_meshes'][name]['local_z_extent_mm']-frames[-1]['surface_meshes'][name]['local_z_extent_mm'])<1e-4
    original=json.loads((ROOT/'preview/contact-surfaces/video-checks.json').read_text())
    for name,change in changes.items():
        assert abs(change/original['actual_surface_relief_changes_mm'][name]-2.5)<.001
    for name,digest in schedule['source_hashes'].items():assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest
    fonts=[ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n) for n in (32,26,20)]
    @lru_cache(maxsize=4)
    def image(index):
        raw=Image.open(OUT/'frames'/f'{index:04d}.png').convert('RGB');assert raw.size==(1200,1000)
        assert all(min(raw.getpixel(p))>=250 for p in ((0,0),(1199,0),(0,999),(1199,999)))
        canvas=Image.new('RGB',(1200,1200),'white');canvas.paste(raw,(0,112));draw=ImageDraw.Draw(canvas)
        f=frames[index]
        draw.text((48,24),'Contact surface geometry',font=fonts[0],fill='#18202a')
        draw.text((48,68),'Height, width, aspect ratio, position and orientation',font=fonts[1],fill='#4a515b')
        draw.line((48,108,1152,108),fill='#e7e9eb')
        draw.text((48,1125),'Original palm and joint-pad geometry | Independent Gaussian bumps',font=fonts[2],fill='#626a74')
        draw.text((48,1158),f"Palm height parameter: {f['palm_surface']['height_mm']:.1f} mm | Pad height parameter: 0-10 mm",font=fonts[2],fill='#626a74')
        bar=1200/schedule['camera_scale']*.02;draw.line((1040,1142,1040+bar,1142),fill='#626a74',width=2)
        for x in (1040,1040+bar):draw.line((x,1137,x,1147),fill='#626a74',width=2)
        draw.text((1038,1154),'20 mm',font=fonts[2],fill='#626a74')
        return canvas
    video=OUT/'contact-surfaces-stronger-5s.mp4'
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
    checks=dict(seconds=5,frames=150,native_renders=101,animated_pad_meshes=len(names),intensity_multiplier=2.5,verified_relief_multiplier='passed',
                actual_surface_relief_changes_mm=changes,closed_surface_loop='passed',fixed_bases_and_angles='passed',
                framing='passed',full_decode='passed',previous_videos_unchanged=True,source_hashes='passed',bytes=video.stat().st_size)
    (OUT/'video-checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'manifest.json').write_text(json.dumps({**schedule,'frames':frames,'preserved_videos':preserved,
        'engine':'BLENDER_WORKBENCH','materials':'Original components.blend','light':'mylight.sl',
        'cavity':{'type':'BOTH','world':[2.5,2.5],'screen':[2,2]}},indent=2))
    shutil.copyfile(ROOT/'preview/thumb-orbit/mylight.sl',OUT/'mylight.sl')
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Contact surface geometry</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#18202a;background:white}h1{font-size:28px}video{display:block;width:100%;max-height:78vh}a{color:#245889}</style><h1>Contact surface geometry</h1><p>Palm and joint-pad contact surfaces vary independently in height, width, aspect ratio, position and orientation, using the original generator Gaussian displacement. A fixed oblique view shows the surface relief.</p><video controls loop preload="metadata" poster="poster.png" src="contact-surfaces-stronger-5s.mp4"></video><p><a href="contact-surfaces-stronger-5s.mp4" download>Download the 5-second video</a></p></html>''',encoding='utf-8')
    (OUT/'render-notes.txt').write_text('Native generation_v2 Gaussian bump displacement on the palm and finger/thumb joint pads. Parameters vary independently per digit and pad: height, spread, aspect ratio, orientation and center position. The native Blender displacement currently evaluates Gaussian kernels regardless of the stored bump_type value, so all types are explicitly Gaussian. Palm resolution 6 and digit resolution 5 resolve the relief. Two bumps per surface. Heights are 2.5 times the prior clip: palm maximum 30 mm and joint maximum 10 mm, with separate phases and a smooth rise/fall envelope. Fixed oblique orthographic camera with projected world-Y up, smooth pad shading, and original geometry, materials and cavity lighting. Height values describe native amplitude parameters; overlapping bumps can produce greater combined relief. 101 native renders sampled into 150 output frames at 30 fps, 5 seconds. Previous videos preserved. Reproduce with build_stronger_surfaces.py.\n')
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
