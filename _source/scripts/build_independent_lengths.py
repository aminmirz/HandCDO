"""Render the approved hand with independent native link lengths on every finger and thumb."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,subprocess
from concurrent.futures import ThreadPoolExecutor,as_completed
from functools import lru_cache
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/independent-link-lengths'
BLENDER='C:/Program Files/Blender Foundation/Blender 5.2/blender.exe'

def plan():
    prior=json.loads((ROOT/'preview/thumb-assembly/manifest.json').read_text())
    final=prior['frames'][-1]
    base={k:final[k] for k in ('thumb_location','locations','identities')}
    # Slots are ordered from the base toward the fingertip in each native chain.
    profiles={1:[(8,0,22,62),(12,7,45,84),(10,19,64,89)],
              2:[(11,4,33,70),(7,15,56,88),(13,0,18,53)],
              3:[(6,13,42,79),(14,0,28,66),(9,9,61,86)],
              0:[(17,3,38,82),(21,17,67,89)]}
    def pulse(i,profile):
        maximum,start,peak,end=profile
        if i<=start or i>=end:return 0.
        t=(i-start)/(peak-start) if i<=peak else (end-i)/(end-peak)
        return maximum*(1-math.cos(math.pi*t))/2
    frames=[]
    for i in range(90):
        lengths={identity:[pulse(i,p) for p in ps] for identity,ps in profiles.items()}
        frames.append(dict(base,index=i,stage=1,finger_link_lengths_mm=[lengths[k] for k in base['identities']],
                           thumb_link_lengths_mm=lengths[0]))
    order=list(range(90))
    schedule=dict(frames=frames,order=order,keyframes=[0,22,45,67,89],fps=30,duration_seconds=3,
                  source_hashes=prior['source_hashes'],baseline=final,profiles=profiles)
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
    requested=schedule['keyframes'] if args.preview else range(90)
    missing=[i for i in requested if not (OUT/'frames'/f'{i:04d}.json').exists()]
    batches=[missing[i:i+15] for i in range(0,len(missing),15)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in as_completed([pool.submit(render,b) for b in batches]):result.result()
    if args.preview:return
    frames=[json.loads((OUT/'frames'/f'{i:04d}.json').read_text()) for i in range(90)]
    baseline=schedule['baseline'];base=np.array(baseline['bases_mm']);normals=np.array(baseline['normals'])
    for f in frames:
        assert f['thumb_count']==1 and f['finger_count']==3 and f['thumb_code']=='1--22' and f['finger_code']=='2-1-1'
        assert np.allclose(f['bases_mm'],base,atol=1e-6) and np.allclose(f['normals'],normals)
        assert np.allclose(f['thumb_base_mm'],baseline['thumb_base_mm']) and np.allclose(f['thumb_normal'],baseline['thumb_normal'])
        lengths=f['finger_link_lengths_mm']+[f['thumb_link_lengths_mm']]
        for requested,actual in zip(lengths,f['native_link_lengths_mm']):
            assert np.allclose([v for v in requested if v>0],actual)
        directions=np.array([list(n)+[0.] for n in normals]+[list(baseline['thumb_normal'])+[0.]])
        expected_tips=np.array(frames[0]['native_tip_positions_mm'])+np.array([sum(v) for v in lengths])[:,None]*directions
        assert np.allclose(f['native_tip_positions_mm'],expected_tips,atol=1e-5)
        assert all(.025<v<.975 for v in f['bounds'])
    tracks=np.array([sum(f['finger_link_lengths_mm'],[])+f['thumb_link_lengths_mm'] for f in frames])
    assert tracks.shape==(90,11) and np.allclose(tracks[0],0) and np.allclose(tracks[-1],0)
    assert len(set(np.argmax(tracks,axis=0)))==11
    assert len(set(np.max(tracks,axis=0)))==11
    rates=np.diff(tracks,axis=0)
    assert any(np.any(r>0) and np.any(r<0) for r in rates)
    for name,digest in schedule['source_hashes'].items():assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest
    fonts=[ImageFont.truetype('C:/Windows/Fonts/arial.ttf',n) for n in (32,26,20)]
    @lru_cache(maxsize=4)
    def image(index):
        raw=Image.open(OUT/'frames'/f'{index:04d}.png').convert('RGB');assert raw.size==(1200,1000)
        assert all(min(raw.getpixel(p))>=250 for p in ((0,0),(1199,0),(0,999),(1199,999)))
        canvas=Image.new('RGB',(1200,1200),'white');canvas.paste(raw,(0,112));draw=ImageDraw.Draw(canvas)
        f=frames[index]
        draw.text((48,24),'Independent link lengths',font=fonts[0],fill='#18202a')
        draw.text((48,68),'Independent added lengths for each link and digit',font=fonts[1],fill='#4a515b')
        draw.line((48,108,1152,108),fill='#e7e9eb')
        def values(lengths):return ' / '.join(f'{v:.1f}' for v in lengths)
        by_id=dict(zip(f['identities'],f['finger_link_lengths_mm']))
        draw.text((48,1125),f"Left: {values(by_id[1])} mm | Middle: {values(by_id[2])} mm",font=fonts[2],fill='#626a74')
        draw.text((48,1158),f"Right: {values(by_id[3])} mm | Thumb: {values(f['thumb_link_lengths_mm'])} mm",font=fonts[2],fill='#626a74')
        bar=1200/.54*.02;draw.line((1040,1142,1040+bar,1142),fill='#626a74',width=2)
        for x in (1040,1040+bar):draw.line((x,1137,x,1147),fill='#626a74',width=2)
        draw.text((1038,1154),'20 mm',font=fonts[2],fill='#626a74')
        return canvas
    video=OUT/'independent-link-lengths-3s.mp4'
    process=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1200x1200','-r','30','-i','-',
             '-an','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
    try:
        for index in schedule['order']:process.stdin.write(image(index).tobytes())
        process.stdin.close();assert process.wait()==0
    finally:
        if process.poll() is None:process.kill()
    image(45).save(OUT/'poster.png')
    sheet=Image.new('RGB',(1200,1200),'white')
    for n,i in enumerate((0,22,45,67)):
        sheet.paste(image(i).resize((600,600),Image.Resampling.LANCZOS),((n%2)*600,(n//2)*600))
    sheet.save(OUT/'stages.png')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(video)],text=True))
    stream=probe['streams'][0]
    assert int(stream['nb_frames'])==90 and stream['r_frame_rate']=='30/1' and abs(float(probe['format']['duration'])-3)<.01
    assert stream['codec_name']=='h264' and stream['pix_fmt']=='yuv420p' and (stream['width'],stream['height'])==(1200,1200)
    subprocess.run(['ffmpeg','-v','error','-xerror','-i',str(video),'-f','null','-'],check=True)
    for name,digest in preserved.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
    checks=dict(seconds=3,frames=90,native_renders=90,independent_link_slots=11,
                distinct_peak_times='passed',distinct_maximum_lengths='passed',native_link_lengths='passed',
                native_tip_displacements='passed',fixed_bases_and_angles='passed',
                framing='passed',full_decode='passed',previous_videos_unchanged=True,source_hashes='passed',bytes=video.stat().st_size)
    (OUT/'video-checks.json').write_text(json.dumps(checks,indent=2))
    (OUT/'manifest.json').write_text(json.dumps({**schedule,'frames':frames,'preserved_videos':preserved,
        'engine':'BLENDER_WORKBENCH','materials':'Original components.blend','light':'mylight.sl',
        'cavity':{'type':'BOTH','world':[2.5,2.5],'screen':[2,2]}},indent=2))
    shutil.copyfile(ROOT/'preview/thumb-orbit/mylight.sl',OUT/'mylight.sl')
    (OUT/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Independent link lengths</title><style>body{max-width:900px;margin:40px auto;padding:0 24px;font:16px/1.6 Arial,sans-serif;color:#18202a;background:white}h1{font-size:28px}video{display:block;width:100%;max-height:78vh}a{color:#245889}</style><h1>Independent link lengths</h1><p>Each finger has three independent added-link lengths, and the thumb has two. Each slot uses a different maximum length and timing. Base positions and angles stay fixed.</p><video controls loop preload="metadata" poster="poster.png" src="independent-link-lengths-3s.mp4"></video><p><a href="independent-link-lengths-3s.mp4" download>Download the 3-second video</a></p></html>''',encoding='utf-8')
    (OUT/'render-notes.txt').write_text('Original generation_v2 thumb 1--22 and three fingers 2-1-1. Uses native FingerConfig.link_added_length_mm_list, with three slots per finger and two per thumb. All eleven slots have distinct maxima (6 to 21 mm) and peak frames, independent staggered smooth pulses, and zero added length at both ends. No digit geometry is replaced; native chain construction, assembly meshes and material assignments are retained. Base positions, angles, palm and thumb mount remain fixed. Original materials, white background, front orthographic camera, mylight.sl, Both cavity world 2.5/2.5 and screen 2/2. 3 seconds at 30 fps. Previous videos preserved. Reproduce using build_independent_lengths.py.\n')
    print(json.dumps(checks,indent=2),flush=True)

if __name__=='__main__':main()
