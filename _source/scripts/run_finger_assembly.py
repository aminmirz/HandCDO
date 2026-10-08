"""Render short independent Blender batches to avoid long-session slowdown."""
import sys
sys.dont_write_bytecode=True
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/finger-assembly'
BLENDER=Path('C:/Program Files/Blender Foundation/Blender 5.2/blender.exe')

def run(indices):
    logs=OUT/'logs';logs.mkdir(exist_ok=True)
    temp=OUT/'tmp'/f'process-{indices[0]:04d}';temp.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',TEMP=str(temp),TMP=str(temp))
    command=[str(BLENDER),'-b','--factory-startup','-t','4','--python-exit-code','1',
             '--python',str(ROOT/'scripts/render_finger_assembly.py'),'--','--output',str(OUT),'--indices',*map(str,indices)]
    log=logs/f'batch-{indices[0]:04d}.log'
    with log.open('w',encoding='utf-8') as output:
        result=subprocess.run(command,env=env,stdout=output,stderr=subprocess.STDOUT,
                              creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError(f'{log.name} failed:\n'+log.read_text(encoding='utf-8',errors='replace')[-4000:])
    print(f'BATCH_COMPLETE {indices[0]}-{indices[-1]}',flush=True)

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output.resolve()
    if not OUT.is_relative_to(ROOT):raise ValueError('Output must remain inside website_assets.')
    schedule=json.loads((OUT/'plan.json').read_text())
    missing=[]
    for f in schedule['frames']:
        image=OUT/'frames'/f'{f["index"]:04d}.png';meta=image.with_suffix('.json')
        cached=json.loads(meta.read_text()) if meta.is_file() else {}
        if not (image.is_file() and cached.get('palette_identity')==['t','f3','f2','f1'] and cached.get('normal_offsets_mm',[])==f.get('normal_offsets_mm',[])):
            missing.append(f['index'])
    batches=[missing[i:i+18] for i in range(0,len(missing),18)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(run,b) for b in batches]):future.result()
    records=[json.loads((OUT/'frames'/f'{f["index"]:04d}.json').read_text()) for f in schedule['frames']]
    manifest={**schedule,'frames':records,'engine':'BLENDER_WORKBENCH',
              'materials':'Original components.blend materials, assigned consistently to persistent finger identities',
              'camera':{'position':[0,0,1],'scale_m':.54,'front_axis':'-Z'},
              'cavity':{'type':'BOTH','world':[2.5,2.5],'screen':[2.,2.]},'light':'mylight.sl',
              'generation_mode':'finger_only','palm_size_mm':130.,'thumb_count':0,
              'note':schedule.get('note','One full boundary circuit, then add fingers 2, 3 and 4. Finger 1 stays on the lower right; the others settle along the top.')}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('ALL_FRAMES_COMPLETE',flush=True)

if __name__=='__main__':main()
