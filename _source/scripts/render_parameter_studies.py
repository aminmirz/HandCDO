"""Render original generation_v2 parameter sweeps with Cycles (run in Blender).

The generator is read-only. Geometry is regenerated at every continuous sample;
discrete topology changes are rendered separately, never mesh-morphed.
All output, temporary files, and caches belong to the supplied render workspace.
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
import random
import tempfile
import time

STUDIES = {
    'finger-location': ('Finger placement', 'Finger locations along the valid palm boundary'),
    'normal-offset': ('Finger base offset', 'Normal offset of the finger bases'),
    'palm-size': ('Palm size', 'Palm width with unchanged finger link dimensions'),
    'base-angle': ('Finger base angle', 'Symmetric rotation of the outer finger bases'),
    'link-length': ('Finger link length', 'Additional length of each finger link'),
    'fingertip-scale': ('Fingertip scale', 'Uniform scale of the finger and thumb tips'),
    'finger-count': ('Number of fingers', 'Discrete configurations on a fixed-width palm'),
    'joint-structure': ('Finger joint structure', 'Discrete kinematic codes with a fixed palm'),
    'joint-motion': ('Joint kinematics', 'Articulation about the original joint axes'),
    'palm-surface': ('Palm contact surface', 'Gaussian deformation of the original palm pad'),
}


def settings_for(study, t):
    p = dict(finger_number=3, thumb_number=1, thumb_side='right',
             palm_size_mm=185., outline_aspect_ratio=1.55,
             copy_finger_settings=False, show_pad_settings=True,
             pad_kernel_count=0, pad_max_intensity=0., pad_resolution=5,
             generate_collision_mesh=False, use_custom_locations=False)
    for i in range(5):
        p.update({f'finger_{i}_rotation': '2', f'finger_{i}_before': '1',
                  f'finger_{i}_after': '1', f'finger_{i}_angle': 0.,
                  f'finger_{i}_normal_offset': 0., f'finger_{i}_side_offset': 0.,
                  f'finger_{i}_link_added_length': 0.,
                  f'finger_{i}_fingertip_scale': [1., 1., 1.],
                  f'finger_{i}_show_pad': True, f'finger_{i}_pad_kernel_count': 0})
    p['thumb_0_fingertip_scale'] = [1., 1., 1.]
    if study == 'finger-location':
        p['use_custom_locations'] = True
        p['thumb_0_location'] = .5
        span = .17 + .16 * t
        for i in range(3): p[f'finger_{i}_location'] = .5 + (i-1)*span
        label = 'Boundary locations t = ' + ', '.join(f'{p[f"finger_{i}_location"]:.2f}' for i in range(3))
    elif study == 'normal-offset':
        value = 22*t
        for i in range(3): p[f'finger_{i}_normal_offset'] = value
        label = f'Finger base normal offset: {value:.1f} mm'
    elif study == 'palm-size':
        p['palm_size_mm'] = 145 + 95*t
        label = f'Palm width: {p["palm_size_mm"]:.0f} mm'
    elif study == 'base-angle':
        value = 24*t
        for i in range(3): p[f'finger_{i}_angle'] = (1-i)*value
        label = f'Outer finger base angles: -{value:.0f} / +{value:.0f} degrees'
    elif study == 'link-length':
        value = 12*t
        for i in range(3): p[f'finger_{i}_link_added_length'] = value
        label = f'Added length per finger link: {value:.1f} mm'
    elif study == 'fingertip-scale':
        value = .75+.6*t
        for i in range(3): p[f'finger_{i}_fingertip_scale'] = [value]*3
        p['thumb_0_fingertip_scale'] = [value]*3
        label = f'Fingertip scale: {value:.2f} x'
    elif study == 'finger-count':
        count = 2 + round(3*t)
        p['finger_number'] = count
        p['palm_size_mm'] = 260.
        label = f'{count} fingers + 1 thumb | Fixed palm width: 260 mm'
    elif study == 'joint-structure':
        choices = [('0', '', '11'), ('2', '1', '1'), ('3', '1', '1'), ('4', '1', '1')]
        rot, before, after = choices[round(t*3)]
        for i in range(3):
            p.update({f'finger_{i}_rotation':rot, f'finger_{i}_before':before, f'finger_{i}_after':after})
        label = f'Finger code: {rot}-{before}-{after}'
    elif study == 'joint-motion':
        label = f'Joint articulation: {t*100:.0f}% of demonstration range'
    elif study == 'palm-surface':
        p.update(show_fingers=False, show_thumbs=False, pad_kernel_count=1,
                 pad_max_intensity=9*t, pad_resolution=7, pad_kernel_0_spread=.22,
                 pad_kernel_0_aspect_ratio=1.6, pad_kernel_0_rotation=25.)
        label = f'Palm surface deformation amplitude: {9*t:.1f} mm'
    return p, label


def orient(obj, target, up=(0,1,0)):
    from mathutils import Vector, Matrix
    z=(obj.location-Vector(target)).normalized()
    x=Vector(up).cross(z).normalized()
    y=z.cross(x).normalized()
    obj.rotation_euler=Matrix((x,y,z)).transposed().to_euler()


def studio(scene, study, width, samples):
    import bpy
    scene.render.engine='CYCLES'
    prefs=bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type='OPTIX'
    prefs.get_devices()
    devices=[]
    for device in prefs.devices:
        device.use=device.type=='OPTIX'
        if device.use: devices.append(device.name)
    if not devices: raise RuntimeError('No OptiX GPU available; refusing silent CPU fallback.')
    scene.cycles.device='GPU'
    scene.cycles.samples=samples
    scene.cycles.use_denoising=True
    scene.cycles.seed=31
    scene.cycles.use_animated_seed=False
    scene.cycles.max_bounces=6
    scene.render.resolution_x=width
    scene.render.resolution_y=round(width*.75)
    scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'
    scene.render.image_settings.color_mode='RGB'
    scene.render.film_transparent=False
    scene.render.fps=24
    scene.view_settings.view_transform='Standard'
    scene.view_settings.look='None'
    scene.view_settings.exposure=0
    scene.view_settings.gamma=1
    world=bpy.data.worlds.new('Study white background')
    world.use_nodes=True
    n=world.node_tree.nodes; n.clear()
    out=n.new('ShaderNodeOutputWorld')
    lighting=n.new('ShaderNodeBackground'); lighting.inputs['Color'].default_value=(1,1,1,1); lighting.inputs['Strength'].default_value=.035 if study=='palm-surface' else .35
    white=n.new('ShaderNodeBackground'); white.inputs['Color'].default_value=(1,1,1,1); white.inputs['Strength'].default_value=1
    rays=n.new('ShaderNodeLightPath'); mix=n.new('ShaderNodeMixShader')
    links=world.node_tree.links
    links.new(rays.outputs['Is Camera Ray'],mix.inputs[0]); links.new(lighting.outputs[0],mix.inputs[1]); links.new(white.outputs[0],mix.inputs[2]); links.new(mix.outputs[0],out.inputs[0])
    scene.world=world
    camera=bpy.data.objects.new('Study camera',bpy.data.cameras.new('Study camera'))
    scene.collection.objects.link(camera); scene.camera=camera
    # World Y is the palm-to-fingertip direction in the original model.
    target=(.035,.04,.022)
    direction=(.28,-.20,.95)
    scale=.57
    if study=='palm-surface': target=(.004,0,.030); direction=(.35,-.60,.72); scale=.285
    from mathutils import Vector
    camera.location=Vector(target)+Vector(direction).normalized()*.9
    camera.data.type='ORTHO'; camera.data.ortho_scale=scale
    orient(camera,target)
    lights=[('Key',(-.32,.32,.58),.8,.45),('Fill',(.35,.05,.38),.3,.35),('Rim',(-.18,-.25,.30),.4,.3)]
    if study=='palm-surface': lights=[('Key',(-.28,.08,.10),2.0,.08),('Fill',(.3,-.1,.3),.06,.3)]
    for name,loc,energy,size in lights:
        light=bpy.data.lights.new('Study '+name,'AREA');light.energy=energy;light.shape='DISK';light.size=size
        obj=bpy.data.objects.new('Study '+name,light);scene.collection.objects.link(obj);obj.location=loc;orient(obj,(0,.04,0))
    return devices


def main():
    import bpy
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--study',choices=list(STUDIES),required=True)
    parser.add_argument('--steps',type=int,default=49)
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--samples',type=int,default=64)
    parser.add_argument('--preview',action='store_true')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    work=args.workspace.resolve();generator=work/'generation_v2'
    # Limit this runner to the user-authorized workspaces.
    local=Path(__file__).resolve().parents[1]
    remote=Path('/home/amin/isaac/RANDOM').resolve()
    if not (work.is_relative_to(local) or work.is_relative_to(remote)):
        raise ValueError('Workspace must be website_assets or /home/amin/isaac/RANDOM.')
    sys.path.insert(0,str(work/'dependencies'))
    import numpy as np
    import scipy,pyclipper
    out=work/('previews' if args.preview else 'renders')/args.study
    out.mkdir(parents=True,exist_ok=True)
    (out/'manifest.json').unlink(missing_ok=True)
    temp=work/'tmp'/args.study;temp.mkdir(parents=True,exist_ok=True)
    tempfile.tempdir=str(temp);os.environ['TMPDIR']=str(temp)
    bpy.context.preferences.filepaths.temporary_directory=str(temp)
    bpy.context.preferences.filepaths.save_version=0
    for obj in list(bpy.data.objects):bpy.data.objects.remove(obj,do_unlink=True)
    spec=importlib.util.spec_from_file_location('study_addon',generator/'blender/HandGeneratorV2.py')
    addon=importlib.util.module_from_spec(spec);sys.modules[spec.name]=addon;spec.loader.exec_module(addon);addon.register()
    settings=bpy.context.scene.handgen_settings
    settings.auto_update=False;settings.generation_v2_dir=str(generator);settings.save_output_dir=str(out)
    devices=[]
    count=4 if args.study in ('finger-count','joint-structure') else args.steps
    indexes=sorted(set([0,count//2,count-1])) if args.preview else range(count)
    frames=[];rest=None
    from mathutils import Matrix,Vector
    for i in indexes:
        start=time.monotonic()
        # Cosine easing slows the sweep at each endpoint; palindrome loops are seamless.
        t=i/(count-1) if count==4 else (1-math.cos(math.pi*i/(count-1)))/2
        params,label=settings_for(args.study,t)
        if args.study!='joint-motion' or rest is None:
            random.seed(42);np.random.seed(42)
            for key,value in params.items():setattr(settings,key,value)
            if addon.generate_hand(bpy.context)!={'FINISHED'}:raise RuntimeError('Original generation failed')
            if bpy.data.objects.get('PalmBody_root') is None:raise RuntimeError('Missing original palm')
            if args.study=='palm-surface':
                for obj in bpy.context.scene.objects:
                    if obj.type=='MESH' and obj.name.startswith('viz_pad_PalmBody'):
                        for polygon in obj.data.polygons:
                            polygon.use_smooth=polygon.normal.z>.3
            # Assembly deliberately removes non-hand objects, including studio lights.
            devices=studio(bpy.context.scene,args.study,args.width,args.samples)
            rest=[(bpy.data.objects[j.link_frame_name],bpy.data.objects[j.link_frame_name].matrix_basis.copy(),Vector((j.axis_x,j.axis_y,j.axis_z)),j.min_val,j.max_val) for j in bpy.context.scene.handgen_joints]
        if args.study=='joint-motion':
            for obj,matrix,axis,lo,hi in rest:
                angle=max(lo,min(hi,.65))*t
                obj.matrix_basis=matrix@Matrix.Rotation(angle,4,axis)
        bpy.context.view_layer.update()
        visible=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.name.startswith('viz_') and not o.hide_render]
        # Record projected bounds to verify framing before publishing the videos.
        from bpy_extras.object_utils import world_to_camera_view
        points=[world_to_camera_view(bpy.context.scene,bpy.context.scene.camera,o.matrix_world@Vector(v)) for o in visible for v in o.bound_box]
        bounds=[min(p.x for p in points),min(p.y for p in points),max(p.x for p in points),max(p.y for p in points)]
        bpy.context.scene.render.filepath=str(out/f'{i:04d}.png')
        bpy.ops.render.render(write_still=True)
        frame={'index':i,'t':t,'label':label,'parameters':params,'bounds':bounds,'visible_meshes':len(visible),'seconds':round(time.monotonic()-start,2)}
        if args.study=='palm-surface':
            pad=bpy.data.objects.get('viz_pad_PalmBody')
            frame['pad_local_z_range_mm']=[min(v.co.z for v in pad.data.vertices)*1000,max(v.co.z for v in pad.data.vertices)*1000]
        frames.append(frame)
        (out/'progress.json').write_text(json.dumps(frame,indent=2))
        print('STUDY_FRAME '+json.dumps({'study':args.study,**{k:frame[k] for k in ('index','label','bounds','seconds')}}),flush=True)
        if i==0:bpy.ops.wm.save_as_mainfile(filepath=str(out/'scene.blend'),copy=True)
    hashes={str(p.relative_to(generator)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(generator.rglob('*')) if p.suffix in ('.py','.blend')}
    manifest={'study':args.study,'title':STUDIES[args.study][0],'description':STUDIES[args.study][1],
              'engine':'CYCLES','device':'OPTIX','gpu':devices,'samples':args.samples,'width':args.width,'height':round(args.width*.75),
              'blender':bpy.app.version_string,'scipy':scipy.__version__,'source_hashes':hashes,'frames':frames,
              'presentation':'Fixed orthographic camera; original components.blend material assignments and shader nodes preserved.',
              'materials':'Original generation_v2 materials; no palette substitutions',
              'motion_note':'Kinematic demonstration, not a collision-checked grasp.' if args.study=='joint-motion' else None}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('STUDY_COMPLETE '+args.study,flush=True)


if __name__=='__main__':main()
