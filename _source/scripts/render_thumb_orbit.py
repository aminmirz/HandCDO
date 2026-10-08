"""Render an actual native thumb chain around the entire palm outline.

The caller supplies a full-boundary thumb point, beyond the UI's lower-palm
placement region. All thumb, mount and palm geometry uses unchanged V2 code.
"""
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_finger_location import ROOT,GEN,addon,style
import bpy
import numpy as np
from mathutils import Vector
from Config import PalmConfig,FingerConfig,HandConfig
from HandClass import Hand
import blender_full_assembly as assembly

OUT=ROOT/'preview/thumb-orbit'
APPROVED=json.loads((ROOT/'preview/front-cavity-horizontal-thumb/model/parameters.json').read_text())
THUMB_CODE=f'{APPROVED["thumb_0_rotation"]}--{APPROVED["thumb_0_after"]}'

def palm_config(t):
    cfg=PalmConfig(data='fixed',detailed_viz=False)
    cfg.palm_size_mm=130.;cfg.outline_sides=4;cfg.outline_aspect_ratio=1.55
    cfg.finger_number=0;cfg.thumb_number=1;cfg.generation_mode='standard'
    cfg.finger_points=np.empty((0,2));cfg.finger_bases=np.empty((0,2))
    cfg.finger_bases_normal_vectors=np.empty((0,2))
    cfg.finger_angle_deg_list=[];cfg.finger_base_normal_offset_mm_list=[]
    cfg.finger_base_side_offset_mm_list=[];cfg.finger_location_list=[]
    cfg.thumb_side_list=[];cfg.thumb_location_list=[t]
    cfg.thumb_base_normal_offset_mm_list=[0.];cfg.thumb_base_side_offset_mm_list=[0.]
    # This original angle parameter aligns the thumb's base direction with the
    # outward boundary normal. Native left/right mount construction is retained.
    cfg.thumb_angle_deg_list=[-math.degrees(math.atan2(cfg.thumb_corner_seg1_mm,cfg.thumb_corner_seg2_mm))]
    cfg.bumps_number=0;cfg.bump_max_height_intensity_mm=0.;cfg.pad_resolution_level=4
    cfg.initialize_outline()
    curve=np.vstack([cfg.outline_points,cfg.outline_points[0]])
    cfg.thumb_points=np.array([cfg._curve_point_at(curve,t)])
    return cfg,curve

def plan():
    cfg,curve=palm_config(0.)
    start=cfg._curve_location_of_point(curve,np.array([65.,-18.]))
    frames=[]
    for i in range(181):
        progress=(1-math.cos(math.pi*i/180))/2
        frames.append(dict(index=i,progress=progress,location=(start+progress)%1.))
    schedule=dict(title='Thumb placement',frames=frames,fps=30,duration_seconds=7,
                  order=[0]*15+list(range(181))+[180]*14,
                  source_hashes={str(p.relative_to(GEN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(GEN.rglob('*')) if p.suffix in ('.py','.blend')},
                  code=THUMB_CODE,palm_size_mm=130.,aspect_ratio=1.55,
                  placement='Full closed palm boundary supplied by rendering caller; native thumb/mount geometry retained')
    assert len(schedule['order'])==210
    (OUT/'plan.json').write_text(json.dumps(schedule,indent=2))
    print('THUMB_PLAN '+json.dumps(dict(start=start,frames=181,seconds=7)),flush=True)

def generate(location,temp):
    cfg,_=palm_config(location)
    thumb=FingerConfig(type='thumb',data='fixed',code=THUMB_CODE,id=0)
    hand_cfg=HandConfig();hand_cfg.collision_mesh=False
    hand=Hand(cfg,[],[thumb],hand_cfg,root_dir=str(temp))
    data=hand.save_assembly_data();palm=hand.save_palm_data()
    if not data or not palm:raise RuntimeError('Native thumb/palm generation failed.')
    addon._clear_generated_hand();addon._clear_template_objects()
    appended=addon._append_all_from_blend(str(GEN/'blender/components.blend'))
    fa=assembly.FingerAssembly.from_existing_scene(data,appended)
    fa.run_assembly();assembly.PalmMesh(palm).generate()
    assembly.assign_viz_materials();assembly.apply_origin_offset(palm)
    addon._clear_template_objects()
    assembly.shade_all_meshes_flat()
    addon._apply_hand_visibility(bpy.context)
    assert len(hand.thumbs)==1 and len(hand.fingers)==0
    return dict(point_mm=cfg.thumb_points.tolist()[0],base_mm=cfg.thumb_bases.tolist()[0],
                normal=cfg.thumb_bases_normal_vectors.tolist()[0],side=cfg.thumb_side_list[0],
                thumb_angle_deg=cfg.thumb_angle_deg_list[0],code=thumb.code)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--indices',nargs='+',type=int)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUT.mkdir(parents=True,exist_ok=True)
    if args.plan:plan();return
    schedule=json.loads((OUT/'plan.json').read_text())
    for name,digest in schedule['source_hashes'].items():
        assert hashlib.sha256((GEN/name).read_bytes()).hexdigest()==digest
    indices=args.indices or list(range(181))
    temp=OUT/'tmp'/f'batch-{indices[0]:04d}';temp.mkdir(parents=True,exist_ok=True)
    frames=OUT/'frames';frames.mkdir(exist_ok=True)
    tempfile.tempdir=str(temp);os.environ['TEMP']=os.environ['TMP']=str(temp)
    bpy.context.preferences.filepaths.temporary_directory=str(temp)
    bpy.context.preferences.filepaths.save_version=0
    addon.register();s=bpy.context.scene.handgen_settings;s.auto_update=False
    s.generation_v2_dir=str(GEN);s.save_output_dir=str(temp)
    s.finger_number=0;s.thumb_number=1;s.generate_collision_mesh=False
    s.show_fingers=False;s.show_thumbs=True
    for index in indices:
        image=frames/f'{index:04d}.png';meta=image.with_suffix('.json')
        if image.exists() and meta.exists() and json.loads(meta.read_text()).get('code')==THUMB_CODE:continue
        f=schedule['frames'][index];start=time.monotonic()
        geometry=generate(f['location'],temp)
        scene=bpy.context.scene;style(scene)
        scene.camera.location=(0,0,1);scene.camera.data.ortho_scale=.54
        bpy.context.view_layer.update()
        hand=bpy.data.collections.get('hand')
        assert hand is not None and any(c.name.startswith('thumb_') for c in hand.children)
        assert not any(c.name.startswith('finger_') for c in hand.children)
        visible=[o for o in scene.objects if o.type=='MESH' and o.name.lower().startswith('viz_') and not o.hide_render]
        from bpy_extras.object_utils import world_to_camera_view
        points=[world_to_camera_view(scene,scene.camera,o.matrix_world@Vector(v)) for o in visible for v in o.bound_box]
        bounds=[min(p.x for p in points),min(p.y for p in points),max(p.x for p in points),max(p.y for p in points)]
        assert all(.025<v<.975 for v in bounds),bounds
        scene.render.filepath=str(image);bpy.ops.render.render(write_still=True)
        result={**f,**geometry,'bounds':bounds,'seconds':round(time.monotonic()-start,2),
                'thumb_count':1,'finger_count':0,'engine':'BLENDER_WORKBENCH'}
        meta.write_text(json.dumps(result,indent=2))
        if index in (0,60,90,120,180):bpy.ops.wm.save_as_mainfile(filepath=str(OUT/f'thumb-{index:04d}.blend'),copy=True)
        print('THUMB_FRAME '+json.dumps(dict(index=index,side=geometry['side'],seconds=result['seconds'])),flush=True)

if __name__=='__main__':main()
