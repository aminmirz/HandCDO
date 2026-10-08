"""Original generator finger-only mode: orbit, then add and place three fingers."""
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_finger_location import ROOT, GEN, BASE, addon, style
import bpy
import numpy as np
from Config import PalmConfig
from mathutils import Vector

OUT=ROOT/'preview/finger-assembly'

def config(locations):
    cfg=PalmConfig(data='fixed',detailed_viz=False)
    cfg.palm_size_mm=130.;cfg.outline_sides=4;cfg.outline_aspect_ratio=1.55
    cfg.generation_mode='finger_only';cfg.finger_number=len(locations);cfg.thumb_number=0
    cfg.finger_location_list=list(locations)
    cfg.finger_angle_deg_list=[0.]*len(locations)
    cfg.finger_base_normal_offset_mm_list=[0.]*len(locations)
    cfg.finger_base_side_offset_mm_list=[0.]*len(locations)
    cfg.update()
    assert np.allclose(cfg.finger_location_list,sorted(locations),atol=1e-6)
    return cfg

def plan():
    cfg=config([0.])
    curve=cfg._region_curve(True)
    targets={name:cfg._curve_location_of_point(curve,np.array(point)) for name,point in
             [('park',(65,-25)),('spawn',(65,20)),('left',(-45,130/1.55/2)),
              ('center',(0,130/1.55/2)),('right',(45,130/1.55/2))]}
    frames=[]
    def add(stage,identity_locations,progress):
        pairs=sorted(identity_locations.items(),key=lambda p:p[1])
        locations=[v for _,v in pairs]
        c=config(locations)
        minimum=None if len(locations)==1 else min(float(np.linalg.norm(a-b)) for i,a in enumerate(c.finger_points) for b in c.finger_points[i+1:])
        assert minimum is None or minimum>=40.5
        frames.append(dict(index=len(frames),stage=stage,progress=progress,
                           identities=[identity for identity,_ in pairs],locations=locations,
                           planned_bases_mm=c.finger_points.tolist(),minimum_spacing_mm=minimum))
    for i in range(91):
        t=(1-math.cos(math.pi*i/90))/2
        add(1,{0:(targets['park']+t)%1.},t)
    fixed={0:targets['park']}
    for identity,target in enumerate(('left','center','right'),start=1):
        for i in range(31):
            t=(1-math.cos(math.pi*i/30))/2
            moving=targets['spawn']+(targets[target]-targets['spawn'])*t
            add(identity+1,{**fixed,identity:moving},t)
        fixed[identity]=targets[target]
    assert len(frames)==184
    schedule=dict(frames=frames,targets=targets,fps=30,duration_seconds=7,
                  order=[0]*6+list(range(184))+[183]*20,
                  source_hashes={str(p.relative_to(GEN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(GEN.rglob('*')) if p.suffix in ('.py','.blend')})
    assert len(schedule['order'])==210
    (OUT/'plan.json').write_text(json.dumps(schedule,indent=2))
    print('ASSEMBLY_PLAN '+json.dumps(dict(targets=targets,frames=len(frames),minimum_spacing_mm=min(f['minimum_spacing_mm'] for f in frames if f['minimum_spacing_mm'] is not None))),flush=True)

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--indices',nargs='+',type=int)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUT=args.output.resolve()
    if not OUT.is_relative_to(ROOT):raise ValueError('Output must remain inside website_assets.')
    OUT.mkdir(parents=True,exist_ok=True)
    if args.plan:plan();return
    schedule=json.loads((OUT/'plan.json').read_text())
    for name,digest in schedule['source_hashes'].items():
        assert hashlib.sha256((GEN/name).read_bytes()).hexdigest()==digest
    indices=args.indices or list(range(len(schedule['frames'])))
    temp=OUT/'tmp'/f'batch-{indices[0]:04d}';temp.mkdir(parents=True,exist_ok=True)
    frames=OUT/'frames';frames.mkdir(exist_ok=True)
    tempfile.tempdir=str(temp);os.environ['TEMP']=os.environ['TMP']=str(temp)
    bpy.context.preferences.filepaths.temporary_directory=str(temp)
    bpy.context.preferences.filepaths.save_version=0
    addon.register()
    settings=bpy.context.scene.handgen_settings
    settings.auto_update=False;settings.generation_v2_dir=str(GEN);settings.save_output_dir=str(temp)
    for key,value in BASE.items():setattr(settings,key,value)
    settings.generation_mode='finger_only';settings.thumb_number=0
    settings.generate_collision_mesh=False;settings.use_custom_locations=True
    for i in range(4):
        for key,value in dict(rotation='2',before='1',after='1',angle=0.,normal_offset=0.,side_offset=0.,
                              link_added_length=0.,fingertip_scale=[1.,1.,1.],show_pad=True,pad_kernel_count=0).items():
            setattr(settings,f'finger_{i}_{key}',value)
    for index in indices:
        f=schedule['frames'][index]
        image=frames/f'{index:04d}.png';meta=frames/f'{index:04d}.json'
        if image.exists() and meta.exists():
            cached=json.loads(meta.read_text())
            if cached.get('palette_identity')==['t','f3','f2','f1'] and cached.get('normal_offsets_mm',[])==f.get('normal_offsets_mm',[]):continue
        settings.finger_number=len(f['locations'])
        offsets=f.get('normal_offsets_mm',[0.]*len(f['locations']))
        for i,location in enumerate(f['locations']):
            setattr(settings,f'finger_{i}_location',location)
            setattr(settings,f'finger_{i}_normal_offset',offsets[i])
        start=time.monotonic()
        if addon.generate_hand(bpy.context)!={'FINISHED'}:raise RuntimeError('Native generation failed.')
        data={};exec((temp/'handgen_blender/palm_cfg.py').read_text(),data)
        assert data['generation_mode']=='finger_only'
        assert np.allclose(data['finger_location_list'],f['locations'],atol=1e-6)
        assert np.allclose(data['finger_bases'],f['planned_bases_mm'],atol=1e-3)
        # The generator sorts by boundary location. Keep each persistent finger
        # associated with its original palette material as new fingers arrive.
        palette_names=['t','f3','f2','f1']
        missing=[name for name in palette_names if bpy.data.materials.get(name) is None]
        if missing:
            with bpy.data.libraries.load(str(GEN/'blender/components.blend'),link=False) as (src,dst):
                assert all(name in src.materials for name in missing)
                dst.materials=missing
        hand=bpy.data.collections['hand']
        for col in hand.children:
            if not col.name.startswith('finger_'):continue
            rank=int(col.name.split('_')[-1])-1
            material=bpy.data.materials[palette_names[f['identities'][rank]]]
            for obj in col.objects:
                if obj.type!='MESH':continue
                for slot in obj.material_slots:
                    m=slot.material
                    if m and m.name.startswith('f') and m.name[1:].isdigit():slot.material=material
        scene=bpy.context.scene;style(scene)
        scene.camera.location=(0,0,1);scene.camera.data.ortho_scale=.54
        bpy.context.view_layer.update()
        visible=[o for o in scene.objects if o.type=='MESH' and o.name.lower().startswith('viz_') and not o.hide_render]
        from bpy_extras.object_utils import world_to_camera_view
        points=[world_to_camera_view(scene,scene.camera,o.matrix_world@Vector(v)) for o in visible for v in o.bound_box]
        bounds=[min(p.x for p in points),min(p.y for p in points),max(p.x for p in points),max(p.y for p in points)]
        assert all(.025<v<.975 for v in bounds),bounds
        scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
        result={**f,'bounds':bounds,'bases_mm':data['finger_bases'],'normals':data['finger_bases_normal_vectors'],
                'seconds':round(time.monotonic()-start,2),'engine':'BLENDER_WORKBENCH','palette_identity':palette_names}
        meta.write_text(json.dumps(result,indent=2))
        if index in schedule.get('keyframes',[0,90,121,152,183]):
            bpy.ops.wm.save_as_mainfile(filepath=str(OUT/f'stage-{f["stage"]}-{index:04d}.blend'),copy=True)
        print('ASSEMBLY_FRAME '+json.dumps(dict(index=index,count=len(f['locations']),seconds=result['seconds'])),flush=True)

if __name__=='__main__':main()
