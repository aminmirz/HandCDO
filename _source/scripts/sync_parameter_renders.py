"""Fetch completed Gamma studies and encode them; all local writes are in website_assets."""
import sys
sys.dont_write_bytecode=True
import argparse
import json
from pathlib import Path
import subprocess
import time
from encode_parameter_studies import ROOT,encode
from render_parameter_studies import STUDIES

REMOTE='/home/amin/isaac/RANDOM/handcdo_cycles_20261005'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    raw=ROOT/'.runtime/cycles-render/renders';raw.mkdir(parents=True,exist_ok=True)
    target=ROOT/'site/media/parameters';target.mkdir(parents=True,exist_ok=True)
    done={};start=time.monotonic()
    while True:
        response=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','gamma',f'python3 -B {REMOTE}/scripts/render_status.py'],check=True,capture_output=True,text=True,timeout=45)
        status=json.loads(response.stdout)
        for study in status['completed']:
            if study not in STUDIES:raise ValueError('Unexpected study identifier')
            if study in done:continue
            print('Downloading '+study,flush=True)
            subprocess.run(['scp','-q','-r',f'gamma:{REMOTE}/renders/{study}',str(raw)],check=True,timeout=180)
            done[study]=encode(raw/study,target)
            (ROOT/'preview/render-sync.json').write_text(json.dumps({'completed':list(done),'remote':status},indent=2))
        if len(done)==len(STUDIES):
            (target/'catalog.json').write_text(json.dumps(list(done.values()),indent=2))
            print('ALL_ORIGINAL_MATERIAL_ANIMATIONS_READY',flush=True);break
        if status['batch'].get('failed'):raise RuntimeError('Remote render failures: '+str(status['batch']['failed']))
        if not args.watch:break
        if time.monotonic()-start>5400:raise TimeoutError('Render synchronization exceeded 90 minutes')
        time.sleep(20)

if __name__=='__main__':main()
