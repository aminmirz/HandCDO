"""Render native finger-location changes with the approved front cavity style."""
import sys
sys.dont_write_bytecode=True
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
GEN=ROOT.parent/'generation_v2'
OUT=ROOT/'preview/finger-location'
sys.path.insert(0,str(ROOT/'.runtime/python'))
sys.path.insert(0,str(GEN))
import bpy
import numpy as np
from mathutils import Vector
spec=importlib.util.spec_from_file_location('location_addon',GEN/'blender/HandGeneratorV2.py')
addon=importlib.util.module_from_spec(spec);sys.modules[spec.name]=addon;spec.loader.exec_module(addon)
addon._ensure_matplotlib_mock()
from Config import PalmConfig
from PalmClass import PalmOutline

BASE=json.loads((ROOT/'preview/front-cavity-horizontal-thumb/requested-settings.json').read_text())

def preflight():
    valid=[]
    for n in range(150,481,5):
        span=n/1000
        cfg=PalmConfig(data='fixed',detailed_viz=False)
        for name in ('palm_size_mm','outline_aspect_ratio','outline_sides'):
            setattr(cfg,name,float(BASE[name]) if name!='outline_sides' else BASE[name])
        cfg.finger_number=3;cfg.thumb_number=1;cfg.thumb_side='right'
        cfg.finger_location_list=[.5-span,.5,.5+span];cfg.thumb_location_list=[.5]
        cfg.finger_angle_deg_list=[0.]*3;cfg.thumb_angle_deg_list=[BASE['thumb_0_angle']]
        cfg.finger_base_normal_offset_mm_list=[0.]*3;cfg.finger_base_side_offset_mm_list=[0.]*3
        cfg.thumb_base_normal_offset_mm_list=[0.];cfg.thumb_base_side_offset_mm_list=[0.]
        requested=list(cfg.finger_location_list)
        try:
            cfg.update()
            if not np.allclose(requested,cfg.finger_location_list,atol=1e-7):continue
            PalmOutline(cfg)
            if not np.allclose(cfg.finger_bases_normal_vectors,[[0,1]]*3,atol=1e-6):continue
            gap=min(np.linalg.norm(a-b) for i,a in enumerate(cfg.finger_bases) for b in cfg.finger_bases[i+1:])
            if gap<40.5:continue
            valid.append(dict(span=span,locations=requested,spacing_mm=float(gap),bases_mm=cfg.finger_bases.tolist()))
        except (ValueError,IndexError):continue
    if len(valid)<2:raise RuntimeError('No useful valid placement range on the compact palm.')
    return dict(close=valid[0],wide=valid[-1],valid_samples=len(valid))

def style(scene):
    scene.render.engine='BLENDER_WORKBENCH'
    s=scene.display.shading
    s.light='STUDIO';s.studio_light='mylight.sl';s.color_type='MATERIAL'
    s.show_shadows=False;s.show_cavity=True;s.cavity_type='BOTH'
    s.cavity_ridge_factor=s.cavity_valley_factor=2.5
    s.curvature_ridge_factor=s.curvature_valley_factor=2.
    s.show_object_outline=True;s.object_outline_color=(0,0,0)
    s.show_specular_highlight=True;s.show_xray=False
    s.background_type='VIEWPORT';s.background_color=(1,1,1);s.studiolight_rotate_z=0
    scene.display.render_aa='32'
    scene.render.resolution_x=1200;scene.render.resolution_y=1000
    scene.render.resolution_percentage=100;scene.render.film_transparent=False
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.render.dither_intensity=0
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.view_settings.exposure=0;scene.view_settings.gamma=1
    camera=bpy.data.objects.new('Fixed front camera',bpy.data.cameras.new('Fixed front camera'))
    scene.collection.objects.link(camera);scene.camera=camera
    camera.location=(.060,.072,1);camera.rotation_euler=(0,0,0)
    camera.data.type='ORTHO';camera.data.ortho_scale=.42
    camera.data.clip_start=.001;camera.data.clip_end=10

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--probe',action='store_true')
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUT.mkdir(parents=True,exist_ok=True)
    temp=OUT/'tmp';temp.mkdir(exist_ok=True)
    frames=OUT/'frames';frames.mkdir(exist_ok=True)
    tempfile.tempdir=str(temp);os.environ['TMP']=os.environ['TEMP']=str(temp)
    bpy.context.preferences.filepaths.temporary_directory=str(temp)
    bpy.context.preferences.filepaths.save_version=0
    ranges=preflight()
    (OUT/'placement-range.json').write_text(json.dumps(ranges,indent=2))
    print('PLACEMENT_RANGE '+json.dumps(ranges),flush=True)
    if args.probe:return
    light=next(s for s in bpy.context.preferences.studio_lights if s.name=='mylight.sl')
    shutil.copyfile(light.path,OUT/'mylight.sl')
    addon.register()
    settings=bpy.context.scene.handgen_settings
    settings.auto_update=False;settings.generation_v2_dir=str(GEN);settings.save_output_dir=str(OUT)
    settings.generate_collision_mesh=False
    for key,value in BASE.items():setattr(settings,key,value)
    settings.use_custom_locations=True;settings.thumb_0_location=.5
    records=[]
    for index in range(91):
        if args.preview and index not in (0,45,90):continue
        target=frames/f'{index:04d}.png';meta=frames/f'{index:04d}.json'
        if target.exists() and meta.exists():
            records.append(json.loads(meta.read_text()));continue
        t=(1-math.cos(math.pi*index/90))/2
        span=ranges['close']['span']+(ranges['wide']['span']-ranges['close']['span'])*t
        requested=[.5-span,.5,.5+span]
        for i,v in enumerate(requested):setattr(settings,f'finger_{i}_location',v)
        start=time.monotonic()
        if addon.generate_hand(bpy.context)!={'FINISHED'}:raise RuntimeError('Native generator failed.')
        # Read the original generator's exported configuration to verify that
        # spacing correction did not silently change the requested locations.
        data={}
        exec((temp/'handgen_blender/palm_cfg.py').read_text(),data)
        assert np.allclose(data['finger_location_list'],requested,atol=1e-6)
        bases=np.asarray(data['finger_bases'])
        spacing=min(float(np.linalg.norm(a-b)) for i,a in enumerate(bases) for b in bases[i+1:])
        assert spacing>=40.5-1e-3
        assert np.allclose(data['finger_bases_normal_vectors'],[[0,1]]*3,atol=1e-5)
        assert np.allclose(data['thumb_bases_normal_vectors'],[[1,0]],atol=1e-5)
        scene=bpy.context.scene;style(scene);bpy.context.view_layer.update()
        visible=[o for o in scene.objects if o.type=='MESH' and o.name.lower().startswith('viz_') and not o.hide_render]
        from bpy_extras.object_utils import world_to_camera_view
        projected=[world_to_camera_view(scene,scene.camera,o.matrix_world@Vector(v)) for o in visible for v in o.bound_box]
        bounds=[min(p.x for p in projected),min(p.y for p in projected),max(p.x for p in projected),max(p.y for p in projected)]
        assert all(.025<v<.975 for v in bounds),bounds
        scene.render.filepath=str(target);bpy.ops.render.render(write_still=True)
        record=dict(index=index,locations=requested,spacing_mm=spacing,bases_mm=bases.tolist(),
                    thumb_bases_mm=data['thumb_bases'],bounds=bounds,seconds=round(time.monotonic()-start,2))
        meta.write_text(json.dumps(record,indent=2));records.append(record)
        if index in (0,90):bpy.ops.wm.save_as_mainfile(filepath=str(OUT/f'endpoint-{index:02d}.blend'),copy=True)
        print('LOCATION_FRAME '+json.dumps(record),flush=True)
    manifest=dict(engine='BLENDER_WORKBENCH',materials='Original components.blend viewport colors',
                  light='mylight.sl',cavity=dict(type='BOTH',world=[2.5,2.5],screen=[2.,2.]),
                  camera=dict(position=[.060,.072,1],scale_m=.42,front_axis='-Z'),
                  fixed_parameters=BASE,range=ranges,frames=records,dimensions=[1200,1000],
                  source_hashes={str(p.relative_to(GEN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(GEN.rglob('*')) if p.suffix in ('.py','.blend')},
                  note='Only finger location parameters change. The native generator rebuilds palm attachment geometry at each sample.')
    (OUT/('preview.json' if args.preview else 'manifest.json')).write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
