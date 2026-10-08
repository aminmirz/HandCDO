"""Native generation_v2 palm studies; all outputs stay inside website_assets.

Run in Blender. The palm-only caller clears fixed digit attachment defaults,
then calls the original Config, Hand, and PalmMesh implementations unchanged.
"""
import sys
sys.dont_write_bytecode = True
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

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT.parent / 'generation_v2'
OUT = ROOT / 'preview/palm-parameters'
sys.path.insert(0, str(ROOT / '.runtime/python'))
sys.path.insert(0, str(GEN))
import bpy
import numpy as np
from mathutils import Vector
spec=importlib.util.spec_from_file_location('palm_study_addon',GEN/'blender/HandGeneratorV2.py')
addon=importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)
addon._ensure_matplotlib_mock()
from Config import PalmConfig, HandConfig
from HandClass import Hand
from blender_full_assembly import PalmMesh, assign_viz_materials

BASE = dict(palm_size_mm=130., outline_sides=4, outline_aspect_ratio=1.55,
            outline_longer_axis='x', outline_rotation_deg=0., smoothing_iters=6,
            smoothing_t=.25, resolution_mm=3.)
TITLES = {'shape':'Outline shape', 'size':'Palm size', 'aspect':'Aspect ratio',
          'rotation':'Outline rotation', 'smoothing':'Corner smoothing', 'gallery':'Combined parameters'}

def configurations():
    result = {}
    result['shape'] = [(dict(BASE, outline_sides=n), f'{n} sides') for n in (3,4,5,6,8,12)]
    for study in ('size','aspect','rotation'):
        frames=[]
        for i in range(49):
            t=(1-math.cos(math.pi*i/48))/2
            p=dict(BASE)
            if study=='size':
                p['palm_size_mm']=90+80*t
                label=f'Outline width: {p["palm_size_mm"]:.1f} mm'
            elif study=='aspect':
                # Give both sides of square equal time, with an exact square
                # at the midpoint. X width remains fixed in the native model.
                half_t=(1-math.cos(math.pi*(i if i<=24 else i-24)/24))/2
                p['outline_aspect_ratio']=.5+.5*half_t if i<=24 else 1+1.4*half_t
                label=f'Width / height: {p["outline_aspect_ratio"]:.2f}'
            else:
                p['outline_rotation_deg']=90*t
                label=f'Outline rotation: {p["outline_rotation_deg"]:.1f} degrees'
            frames.append((p,label))
        result[study]=frames
    result['smoothing']=[(dict(BASE, smoothing_iters=n, resolution_mm=10.), f'Smoothing iterations: {n} | t = 0.25') for n in range(9)]
    gallery=[(3,110,1.1,0),(4,130,1.55,0),(5,145,1.2,18),(6,120,1.8,30),
             (8,150,1.0,22.5),(4,110,2.2,45),(6,140,1.5,75),(12,130,1.2,0)]
    result['gallery']=[(dict(BASE,outline_sides=n,palm_size_mm=float(w),outline_aspect_ratio=a,outline_rotation_deg=float(r)),
                        f'{n} sides | {w} mm | ratio {a:.2f} | {r:g} degrees') for n,w,a,r in gallery]
    return result

def generate(parameters, folder):
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj,do_unlink=True)
    # Avoid growing data-blocks through a long rendering session.
    for mesh in list(bpy.data.meshes):
        if mesh.users==0:bpy.data.meshes.remove(mesh)
    cfg=PalmConfig(data='fixed',detailed_viz=False)
    for key,value in parameters.items():setattr(cfg,key,value)
    cfg.finger_number=cfg.thumb_number=0
    # Config's fixed defaults contain digit points. Clear them explicitly for
    # this isolated outline study; no generator files or algorithms are patched.
    for prefix in ('finger','thumb'):
        for suffix in ('points','bases','bases_normal_vectors'):
            setattr(cfg,prefix+'_'+suffix,np.empty((0,2),dtype=float))
        for suffix in ('angle_deg_list','base_normal_offset_mm_list','base_side_offset_mm_list','location_list'):
            setattr(cfg,prefix+'_'+suffix,[])
    cfg.thumb_side_list=[]
    cfg.bumps_number=0
    cfg.bump_max_height_intensity_mm=0.
    cfg.pad_resolution_level=4
    cfg.initialize_outline()
    hand_cfg=HandConfig()
    hand_cfg.collision_mesh=False
    hand=Hand(cfg,[],[],hand_cfg,root_dir=str(folder))
    palm_data=hand.save_palm_data()
    if not palm_data:raise RuntimeError('Native palm generation returned no data.')
    PalmMesh(palm_data).generate()
    if not bpy.data.collections.get('hand'):
        bpy.context.scene.collection.children.link(bpy.data.collections.new('hand'))
    assign_viz_materials()
    visible=[]
    for obj in bpy.context.scene.objects:
        if obj.type=='MESH':
            obj.hide_render=not obj.name.lower().startswith('viz_')
            if not obj.hide_render:
                visible.append(obj)
                for poly in obj.data.polygons:poly.use_smooth=False
    if not visible:raise RuntimeError('No native palm meshes.')
    return visible

def studio(visible, camera_scale):
    scene=bpy.context.scene
    scene.render.engine='BLENDER_WORKBENCH'
    shading=scene.display.shading
    shading.light='STUDIO';shading.studio_light='mylight.sl';shading.color_type='MATERIAL'
    shading.show_shadows=False;shading.show_cavity=True;shading.cavity_type='BOTH'
    shading.cavity_ridge_factor=2.5;shading.cavity_valley_factor=2.5
    shading.curvature_ridge_factor=2.;shading.curvature_valley_factor=2.
    shading.show_object_outline=True;shading.object_outline_color=(0,0,0)
    shading.show_specular_highlight=True;shading.show_xray=False
    shading.background_type='VIEWPORT';shading.background_color=(1,1,1)
    shading.studiolight_rotate_z=0
    scene.display.render_aa='32'
    scene.render.resolution_x=1200;scene.render.resolution_y=1000
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.image_settings.color_mode='RGB'
    scene.render.film_transparent=False;scene.render.dither_intensity=0
    scene.view_settings.view_transform='Standard';scene.view_settings.look='None'
    scene.view_settings.exposure=0;scene.view_settings.gamma=1
    camera=bpy.data.objects.new('Fixed front camera',bpy.data.cameras.new('Fixed front camera'))
    scene.collection.objects.link(camera);scene.camera=camera
    camera.location=(0,0,1);camera.rotation_euler=(0,0,0)
    camera.data.type='ORTHO';camera.data.ortho_scale=camera_scale
    camera.data.clip_start=.001;camera.data.clip_end=10
    bpy.context.view_layer.update()
    from bpy_extras.object_utils import world_to_camera_view
    points=[world_to_camera_view(scene,camera,o.matrix_world@Vector(v)) for o in visible for v in o.bound_box]
    bounds=[min(p.x for p in points),min(p.y for p in points),max(p.x for p in points),max(p.y for p in points)]
    assert all(.04<v<.96 for v in bounds),bounds
    return bounds

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--preview',action='store_true')
    parser.add_argument('--studies',nargs='+',choices=list(TITLES),default=list(TITLES))
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUT.mkdir(parents=True,exist_ok=True)
    temp=OUT/'tmp';temp.mkdir(exist_ok=True)
    tempfile.tempdir=str(temp);os.environ['TEMP']=os.environ['TMP']=str(temp)
    bpy.context.preferences.filepaths.temporary_directory=str(temp)
    bpy.context.preferences.filepaths.save_version=0
    light=next(s for s in bpy.context.preferences.studio_lights if s.name=='mylight.sl')
    shutil.copyfile(light.path,OUT/'mylight.sl')
    with bpy.data.libraries.load(str(GEN/'blender/components.blend'),link=False) as (src,dst):
        dst.materials=list(src.materials)
    studies=configurations()
    for study in args.studies:
        camera_scale=.36 if study=='aspect' else .28
        folder=OUT/'frames'/study;folder.mkdir(parents=True,exist_ok=True)
        frame_meta=[]
        for i,(parameters,label) in enumerate(studies[study]):
            if args.preview and i not in (0,len(studies[study])//2,len(studies[study])-1):continue
            target=folder/f'{i:04d}.png';meta=folder/f'{i:04d}.json'
            if target.exists() and meta.exists():
                saved=json.loads(meta.read_text())
                if saved['parameters']==parameters and saved.get('camera_scale_m',.28)==camera_scale:
                    frame_meta.append(saved);continue
            start=time.monotonic()
            visible=generate(parameters,temp)
            bounds=studio(visible,camera_scale)
            bpy.context.scene.render.filepath=str(target)
            bpy.ops.render.render(write_still=True)
            record=dict(index=i,parameters=parameters,label=label,bounds=bounds,camera_scale_m=camera_scale,meshes=len(visible),seconds=round(time.monotonic()-start,2))
            meta.write_text(json.dumps(record,indent=2))
            frame_meta.append(record)
            if i==0:bpy.ops.wm.save_as_mainfile(filepath=str(folder/'scene.blend'),copy=True)
            print('PALM_PROGRESS '+json.dumps(dict(study=study,**record)),flush=True)
        manifest=dict(study=study,title=TITLES[study],engine='BLENDER_WORKBENCH',
                      dimensions=[1200,1000],camera=f'Fixed front orthographic, scale {camera_scale} m',camera_scale_m=camera_scale,
                      materials='Unchanged original components.blend material viewport colors',
                      cavity=dict(type='BOTH',world_ridge=2.5,world_valley=2.5,screen_ridge=2.,screen_valley=2.),
                      light='mylight.sl',frames=frame_meta,preview=args.preview,
                      source_hashes={str(p.relative_to(GEN)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(GEN.rglob('*')) if p.suffix in ('.py','.blend')},
                      note='Isolated palm outlines without digit mounts. Native Config -> Hand -> PalmMesh; fixed attachment defaults cleared by caller. Rotation retains the generator\'s X-width normalization.')
        (folder/('preview.json' if args.preview else 'manifest.json')).write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
