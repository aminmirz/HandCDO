"""Render the approved hand with outer fingers and thumb rotating together."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/combined-base-angles'
BLENDER='C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'

def plan():
    prior=json.loads((ROOT/'preview/thumb-assembly/manifest.json').read_text())
    final=prior['frames'][-1]
    base={k:final[k] for k in ('thumb_location','locations','identities')}
    frames=[]
    for i in range(37):
        angle=20*(1-math.cos(math.pi*i/36))/2
        angles=[{1:-angle,2:0.,3:angle}[identity] for identity in base['identities']]
        frames.append(dict(base,index=i,stage=1,angles_deg=angles,thumb_angle_delta_deg=-angle,opening_deg=angle))
    order=[0]*6+list(range(37))+list(range(35,-1,-1))+[0]*11
    assert len(frames)==37 and len(order)==90
    schedule=dict(frames=frames,order=order,keyframes=[0,18,36],fps=30,duration_seconds=3,
                  source_hashes=prior['source_hashes'],baseline=final,max_angle_deg=20.)
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
    requested=schedule['keyframes'] if args.preview else range(37)
    missing=[i for i in requested if not (OUT/'frames'/f'{i:04d}.json').exists()]
    batches=[missing[i:i+15] for i in range(0,len(missing),15)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in as_completed([pool.submit(render,b) for b in batches]):result.result()
    if args.preview:return
    frames=[json.loads((OUT/'frames'/f'{i:04d}.json').read_text()) for i in range(37)]
    baseline=schedule['baseline'];base=np.array(baseline['bases_mm']);normals=np.array(baseline['normals'])
    for f in frames:
        angles=np.array(f['angles_deg']);angle=f['opening_deg']
        assert 0<=angle<=20
        assert np.allclose(angles,[{1:-angle,2:0.,3:angle}[identity] for identity in f['identities']])
        assert f['thumb_count']==1 and f['finger_count']==3 and f['thumb_code']=='1--22' and f['finger_code']=='2-1-1'
        assert np.allclose(f['bases_mm'],base,atol=1e-6)
        expected=[]
        for normal,theta in zip(normals,np.radians(angles)):
            expected.append(normal@np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]))
        assert np.allclose(f['normals'],expected,atol=1e-6)
        thumb_expected=[math.cos(math.radians(angle)),-math.sin(math.radians(angle))]
        assert np.allclose(f['thumb_normal'],thumb_expected,atol=1e-6)
        assert all(.025<v<.975 for v in f['bounds'])
    assert frames[-1]['opening_deg']==20 and schedule['order'][0]==schedule['order'][-1]==0
    for name,digest in schedule['source_hashes'].items():assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest
    fonts=[ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n) for n in (32,26,20)]
    @lru_cache(maxsize=4)
    def image(index):
        raw=Image.open(OUT/'frames'/f'{index:04d}.png').convert('RGB');assert raw.size==(1200,1000)
        assert all(min(raw.getpixel(p))>=250 for p in ((0,0),(1199,0),(0,999),(1199,999)))
        canvas=Image.new('RGB',(1200,1200),'white');canvas.paste(raw,(0,112));draw=ImageDraw.Draw(canvas)
        angle=frames[index]['opening_deg']
        draw.text((48,24),'Combined base angles',font=fonts[0],fill='#18202a')
        draw.text((48,68),f'Outer fingers: {angle:.1f} deg outward | Thumb: {angle:.1f} deg clockwise',font=fonts[1],fill='#4a515b')
        draw.line((48,108,1152,108),fill='#e7e9eb')
        draw.text((48,1155),'Thumb: 1--22 | Fingers: 2-1-1 | Palm size: 130 mm',font=fonts[2],fill='#626a74')
        bar=1200/.54*.02;draw.line((1040,1142,1040+bar,1142),fill='#626a74',width=2)
        for x in (1040,1040+bar):draw.line((x,1137,x,1147),fill='#626a74',width=2)
        draw.text((1038,1154),'20 mm',font=fonts[2],fill='#626a74')
        return canvas
    video=OUT/'combined-base-angles-3s.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1200x1200','-r','30','-i','-',
             '-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    try:
        for index in schedule['order']:process.stdin.write(image(index).tobytes())
        process.stdin.close();assert process.wait()==0
    finally:
        if process.poll() is None:process.kill()
    image(36).save(OUT/'poster.png')
    sheet=Image.new('RGB',(1200,1200),'white')
    for n,i in enumerate((0,12,24,36)):
        sheet.paste(image(i).resize((600,600),Image.Resampling.LANCZOS),((n%2)*600,(n//2)*600))
    sheet.save(OUT/'stages.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
    stream=probe['streams'][0]
    assert int(stream['nb_frames'])==90 and stream['r_frame_rate']=='30/1' and abs(float(probe['format']['duration'])-3)<.01
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p' and (stream['width'],stream['height'])==(1200,1200)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    for name,digest in preserved.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    checks=dict(seconds=3,frames=90,native_renders=37,max_outward_angle_deg=20,max_thumb_clockwise_deg=20,
                simultaneous_motion='passed',fixed_middle_finger='passed',native_base_angles='passed',
                framing='passed',full_decode='passed',previous_videos_unchanged=True,source_hashes='passed',bytes=video.stat().st_size)
    (OUT/'video-checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'manifest.json').write_text(json.dumps({**schedule,'frames':frames,'preserved_videos':preserved,
        'engine':'BLENDER_WORKBENCH','materials':'Original components.blend','light':'mylight.sl',
        'cavity':{'type':'BOTH','world':[2.5,2.5],'screen':[2,2]}},indent=2))
    shutil.copyfile(ROOT/'preview/thumb-orbit/mylight.sl',OUT/'mylight.sl')
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Combined base angles</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#18202a;background:white}h1{font-size:28px}video{display:block;width:100%;max-height:78vh}a{color:#245889}</style><h1>Combined base angles</h1><p>The outer fingers tilt outward by 20 degrees while the thumb rotates clockwise by 20 degrees. They move together and return to the starting configuration. The middle finger remains upright.</p><video controls loop preload="metadata" poster="poster.png" src="combined-base-angles-3s.mp4"></video><p><a href="combined-base-angles-3s.mp4" download>Download the 3-second video</a></p></html>''',encoding='utf-8')
    (OUT/'render-notes.txt').write_text('Original generation_v2 thumb 1--22 and three fingers 2-1-1, from the approved thumb-assembly final frame. Native finger_angle_deg_list moves the left finger -20 degrees and right +20 degrees, simultaneously with thumb_angle_deg_list decreased by 20 degrees (clockwise in the front view). Middle finger remains upright; fingers retain base positions. Native thumb mount regenerates with its base-angle parameter. All return to the starting configuration. Original materials, white background, front orthographic camera, mylight.sl, Both cavity world 2.5/2.5 and screen 2/2. 3 seconds at 30 fps; 37 unique native frames. Previous videos preserved. Reproduce using build_combined_angles.py.\n')
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
