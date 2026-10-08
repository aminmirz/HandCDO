"""Build an outward-offset sweep from the approved four-finger configuration."""
import sys
sys.dont_write_bytecode=True
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'preview/finger-offset'
OUT.mkdir(parents=True,exist_ok=True)
approved=json.loads((ROOT/'preview/finger-assembly/manifest.json').read_text())
base=approved['frames'][-1]
for name,digest in approved['source_hashes'].items():
    assert hashlib.sha256((ROOT.parent/'generation_v2'/name).read_bytes()).hexdigest()==digest
frames=[]
for i in range(91):
    offset=25*(1-math.cos(math.pi*i/90))/2
    offsets=[0. if identity==0 else offset for identity in base['identities']]
    points=[[v+off*n for v,n in zip(point,normal)] for point,normal,off in zip(base['bases_mm'],base['normals'],offsets)]
    spacing=min(math.dist(a,b) for j,a in enumerate(points) for b in points[j+1:])
    assert spacing>40.5
    frames.append(dict(index=i,stage=4,progress=i/90,identities=base['identities'],locations=base['locations'],
                       normal_offsets_mm=offsets,offset_mm=offset,planned_bases_mm=points,minimum_spacing_mm=spacing))
schedule=dict(title='Finger base offset',frames=frames,keyframes=[0,45,90],fps=30,duration_seconds=7,
              order=[0]*15+list(range(91))+[90]*15+list(range(89,0,-1)),
              source_hashes=approved['source_hashes'],approved_baseline='preview/finger-assembly/frames/0183.json',
              note='Approved four-finger layout. The three top finger bases move outward along their original normals from 0 to 25 mm and back; the right-side finger remains fixed. Palm attachment geometry is rebuilt by the native generator.')
assert len(schedule['order'])==210
(OUT/'plan.json').write_text(json.dumps(schedule,indent=2))
print('Offset plan: 0 to 25 mm; 91 native samples; 7 seconds; same approved camera and material identities.')
