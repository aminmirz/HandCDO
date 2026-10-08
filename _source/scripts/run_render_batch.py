"""Run the Cycles studies sequentially on Gamma; no global configuration changes."""
import sys
sys.dont_write_bytecode=True
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import threading
from render_parameter_studies import STUDIES

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--workers',type=int,choices=(1,2),default=2)
    parser.add_argument('--studies',nargs='+',choices=list(STUDIES),default=list(STUDIES))
    args=parser.parse_args()
    work=args.workspace.resolve()
    if not work.is_relative_to(Path('/home/amin/isaac/RANDOM')):raise ValueError('Unexpected render workspace')
    logs=work/'logs';logs.mkdir(exist_ok=True)
    prefix='preview' if args.preview else 'final'
    if args.detach:
        with (logs/f'{prefix}-batch.log').open('w') as log:
            process=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),*[a for a in sys.argv[1:] if a!='--detach']],
                cwd=work,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'pid':process.pid,'log':str(logs/f'{prefix}-batch.log')}));return
    env=dict(os.environ,TMPDIR=str(work/'tmp'),XDG_CACHE_HOME=str(work/'cache'),PYTHONDONTWRITEBYTECODE='1',
             CUDA_CACHE_PATH=str(work/'cache'/'cuda'),OPTIX_CACHE_PATH=str(work/'cache'/'optix'),OMP_NUM_THREADS='8')
    state={'completed':[],'failed':[],'current':[],'pid':os.getpid(),'preview':args.preview,'started':time.time()}
    lock=threading.Lock()
    def save():
        (work/f'{prefix}-status.json').write_text(json.dumps(state,indent=2))
    def render(study):
        with lock:
            state['current'].append(study);save()
        cmd=['/snap/bin/blender','-b','-t','8','--factory-startup','--python-exit-code','1','--python',str(work/'scripts'/'render_parameter_studies.py'),
             '--','--workspace',str(work),'--study',study,'--width','800' if args.preview else '1280','--samples','24' if args.preview else '64']
        if args.preview:cmd.append('--preview')
        with (logs/f'{prefix}-{study}.log').open('w') as log:
            result=subprocess.run(cmd,cwd=work,env=env,stdout=log,stderr=subprocess.STDOUT)
        with lock:
            state['completed' if result.returncode==0 else 'failed'].append(study)
            state['current'].remove(study);save()
        print(study,result.returncode,flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(render,args.studies))
    state['current']=None;state['finished']=time.time()
    (work/f'{prefix}-status.json').write_text(json.dumps(state,indent=2))
    if state['failed']:raise RuntimeError('Failed: '+str(state['failed']))

if __name__=='__main__':main()
